import torch
from transformers import BertModel, BertTokenizer
import os
from pathlib import Path
def main():
    device = torch.device("cpu")

    BASE_DIR = Path(__file__).resolve().parent.parent
    path = os.path.join(BASE_DIR, 'model' ,'epoch20.pth')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = torch.load(path, map_location=device)
if __name__ == '__main__':
    main()
