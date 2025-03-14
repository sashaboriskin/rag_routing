from dataset import (
    NQDataset,
    WikiMultiHopQADataset, 
    HotPotQADataset, 
    MusiqueDataset
)

nq_dataset = NQDataset().data
wiki_multi_dataset = WikiMultiHopQADataset().data
hot_pot_dataset = HotPotQADataset().data
musique_dataset = MusiqueDataset().data