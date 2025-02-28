import collections
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from transformers import AutoTokenizer


def get_attributes(x: nn.Module, attributes: str):
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x

def set_attribute_recursive(x: nn.Module, attributes: str, new_attribute: nn.Module):
    parts = attributes.split(".")
    for attr in parts[:-1]:
        x = getattr(x, attr)
    setattr(x, parts[-1], new_attribute)

def get_ff_layer(
    model: nn.Module,
    layer_idx: int,
    transformer_layers_attr: str = "model.layers",
    ff_attrs: str = "mlp.gate_proj",
):

    transformer_layers = get_attributes(model, transformer_layers_attr)
    ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    return ff_layer

def register_hook(model: nn.Module, layer_idx: int, f, transformer_layers_attr: str, ff_attrs: str):
    layer = get_ff_layer(model, layer_idx, transformer_layers_attr, ff_attrs)
    handle = layer.register_forward_hook(lambda module, inp, out: f(out))
    return handle

class MlpPatch(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        mask_idx: int,
        replacement_activations: Optional[torch.Tensor] = None,
        target_positions: Optional[List[int]] = None,
        mode: str = "replace",
        enhance_value: float = 2.0,
    ):
        super().__init__()
        self.layer_function = layer
        self.acts = replacement_activations
        self.mask_idx = mask_idx
        self.target_positions = target_positions
        self.enhance_value = enhance_value
        assert mode in ["replace", "suppress", "enhance"]
        self.mode = mode
        if self.mode == "replace":
            assert self.acts is not None
        elif self.mode in ["enhance", "suppress"]:
            assert self.target_positions is not None

    def forward(self, x: torch.Tensor):
        x = self.layer_function(x)
        
        if self.mode == "replace":
            x[:, self.mask_idx, :] = self.acts
            
        elif self.mode == "suppress":
            for pos in self.target_positions:
                
                x[:, self.mask_idx, pos] = 0.0
        elif self.mode == "enhance":
            for pos in self.target_positions:
                x[:, self.mask_idx, pos] *= self.enhance_value
        
        return x

def mlp_patch_layer(
    model: nn.Module,
    mask_idx: int,
    layer_idx: int,
    replacement_activations: Optional[torch.Tensor] = None,
    mode: str = "replace",
    transformer_layers_attr: str = "model.layers",
    ff_attrs: str = "mlp.gate_proj",
    neurons: Optional[List[List[int]]] = None,
):

    transformer_layers = get_attributes(model, transformer_layers_attr)
    if mode == "replace":
        layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
        assert layer_idx < len(transformer_layers), f"Нет слоя {layer_idx} (всего {len(transformer_layers)})"
        set_attribute_recursive(
            transformer_layers[layer_idx],
            ff_attrs,
            MlpPatch(
                layer,
                mask_idx,
                replacement_activations=replacement_activations,
                mode=mode,
            ),
        )
    elif mode in ["suppress", "enhance"]:
        neurons_dict = collections.defaultdict(list)
        for neuron in neurons:
            # neuron: [layer_idx, pos]
            layer_idx_neuron, pos = neuron
            neurons_dict[layer_idx_neuron].append(pos)
        for layer_idx_neuron, positions in neurons_dict.items():
            layer = get_attributes(transformer_layers[layer_idx_neuron], ff_attrs)
            set_attribute_recursive(
                transformer_layers[layer_idx_neuron],
                ff_attrs,
                MlpPatch(
                    layer,
                    mask_idx,
                    replacement_activations=None,
                    mode=mode,
                    target_positions=positions,
                ),
            )

def mlp_unpatch_layer(
    model: nn.Module,
    layer_idx: int,
    transformer_layers_attr: str = "model.layers",
    ff_attrs: str = "mlp.gate_proj",
):
    transformer_layers = get_attributes(model, transformer_layers_attr)
    layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    set_attribute_recursive(
        transformer_layers[layer_idx],
        ff_attrs,
        layer.layer_function,
    )

def mlp_unpatch_layers(
    model: nn.Module,
    layer_indices,
    transformer_layers_attr: str = "model.layers",
    ff_attrs: str = "mlp.gate_proj",
):
    for layer_idx in layer_indices:
        mlp_unpatch_layer(model, layer_idx, transformer_layers_attr, ff_attrs)


class KnowledgeNeurons:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = tokenizer

        self.baseline_activations = None
        self.transformer_layers_attr = "model.layers"
        self.input_ff_attr = "mlp.gate_proj"

    def prepare_inputs(self, prompt: str, target: Optional[str] = None):
        encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        mask_idx = -1
        
        if target is not None:
            target_label = self.tokenizer.encode(target, add_special_tokens=False)
        else:
            target_label = None
            
        return encoded_input, mask_idx, target_label, prompt

    def scaled_input(self, activations: torch.Tensor, steps: int = 20):

        baseline = torch.zeros_like(activations)
        num_points = steps
        step = (activations - baseline) / num_points
        scaled = torch.cat([baseline + step * i for i in range(num_points)], dim=0)
        return scaled, step

    def get_baseline_with_activations(self, encoded_input, layer_idx: int, mask_idx: int):

        def hook_fn(acts):
            self.baseline_activations = acts[:, mask_idx, :]

        handle = register_hook(
            self.model, 
            layer_idx, 
            hook_fn,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr
        )
        
        baseline_outputs = self.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations

    def get_scores_for_layer(self, prompt: str, ground_truth: str, layer_idx: int,
                             batch_size: int = 10, steps: int = 20):

        encoded_input, mask_idx, target_label, prompt = self.prepare_inputs(prompt, ground_truth)
        n_sampling_steps = len(target_label) if target_label is not None else 1
        integrated_grads = []
        answer = ''
        
        for i in range(n_sampling_steps):
            if i > 0:
                encoded_input, mask_idx, target_label, prompt = self.prepare_inputs(prompt, ground_truth)
                
            baseline_outputs, baseline_activations = self.get_baseline_with_activations(encoded_input, layer_idx, mask_idx)
            baseline_outputs = baseline_outputs.logits.detach().cpu()
            baseline_activations = baseline_activations.detach().cpu()
            
            argmax_next_token = baseline_outputs[:, mask_idx, :].argmax(dim=-1).item()
            next_token_str = self.tokenizer.decode([argmax_next_token])
                
            scaled_weights, weights_step = self.scaled_input(baseline_activations, steps=steps)
            scaled_weights.requires_grad_(True)
            integrated_grads_this_step = []
            n_batches = steps // batch_size
            
            for batch_weights in scaled_weights.chunk(n_batches):
                
                inputs = {
                    "input_ids": einops.repeat(encoded_input["input_ids"], "b d -> (r b) d", r=batch_size),
                    "attention_mask": einops.repeat(encoded_input["attention_mask"], "b d -> (r b) d", r=batch_size),
                }

                mlp_patch_layer(
                    self.model,
                    mask_idx=mask_idx,
                    layer_idx=layer_idx,
                    replacement_activations=batch_weights,
                    mode="replace",
                    transformer_layers_attr=self.transformer_layers_attr,
                    ff_attrs=self.input_ff_attr,
                )
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
                target_idx = target_label[i] if target_label is not None else 0
                grad = torch.autograd.grad(torch.unbind(probs[:, target_idx]), batch_weights)[0]
                grad = grad.sum(dim=0)
                integrated_grads_this_step.append(grad)
         
                mlp_unpatch_layer(
                    self.model,
                    layer_idx=layer_idx,
                    transformer_layers_attr=self.transformer_layers_attr,
                    ff_attrs=self.input_ff_attr,
                )
                del outputs, probs, grad
                torch.cuda.empty_cache()
            
            integrated_grads_this_step = torch.stack(integrated_grads_this_step, dim=0).sum(dim=0)
            integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
            integrated_grads.append(integrated_grads_this_step)
            
            prompt += next_token_str
            answer += next_token_str
                
        integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(integrated_grads)
        return integrated_grads, answer

    def get_coarse_neurons(self, prompt: str, ground_truth: str, layer_idx: int, batch_size: int = 10,
                             steps: int = 20, threshold: Optional[float] = None,
                             adaptive_threshold: Optional[float] = None,
                             percentile: Optional[float] = None) -> List[List[int]]:

        attribution_scores, answer = self.get_scores_for_layer(prompt, ground_truth, layer_idx=layer_idx,
                                                        batch_size=batch_size, steps=steps)
        if adaptive_threshold is not None:
            threshold = attribution_scores.max().item() * adaptive_threshold    
        if threshold is not None and threshold > 0:
            indices = torch.nonzero(attribution_scores > threshold).cpu()#.tolist()
        else:
            s = attribution_scores.flatten().detach().cpu().numpy()
            perc = np.percentile(s, percentile) if percentile is not None else 50
            indices = torch.nonzero(attribution_scores > perc).cpu()#.tolist()

        return indices.flatten().tolist(), answer
