#/**
# * @author مۇختەرجان مەخمۇت
# * @email ug-project@outlook.com
# * @create date 2024-07-28 03:30:26
# * @modify date 2024-07-28 03:30:26
# * @desc [description]
#*/

import torch
import torch.utils
import torch.utils.data
from transformers import BertTokenizer, BertModel, modeling_outputs
import json
import typing
from typing import Any
import transformers


#Classifier neural network composed with Google bert and MetaAI nllb200.
class MyClassifierNet(torch.nn.Module):
    def __init__(self, device: torch.device, output_count: int) -> None:
        super(MyClassifierNet, self).__init__()

        self.device = device

        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_tokenizer: BertTokenizer
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        self.nllb_tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", tgt_lang="eng_Latn", src_lang="uig_Arab")
        self.nllb_model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        self.nllb_model: transformers.AutoModelForSeq2SeqLM

        from torch.nn import Dropout, Linear, ReLU
        self.dropout = Dropout(0.2)
        self.fc0 = Linear(self.bert_model.base_model.config.hidden_size, 512)
        self.activation_function = ReLU()
        self.fc1 = Linear(512, 128)
        self.fc2 = Linear(128, output_count)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        output = self.dropout(output)
        output = self.fc0(output)
        output = self.activation_function(output)
        output = self.fc1(output)
        output = self.activation_function(output)
        output = self.fc2(output)
        return output
    
    def translate(self, text: str) -> str:
        result = self.nllb_tokenizer(text, return_tensors="pt").to(self.device)
        result = self.nllb_model.generate(**result, forced_bos_token_id=self.nllb_tokenizer.added_tokens_encoder["eng_Latn"])
        result = self.nllb_tokenizer.batch_decode(result, skip_special_tokens=True)[0]
        # print(result)
        return result

    
    def predict(self, prompt: str) -> torch.Tensor:
        prompt = self.translate(prompt)
        encoded_input = self.bert_tokenizer.encode_plus(prompt, add_special_tokens=True, truncation=True, padding="max_length", max_length=32, return_tensors="pt").to(self.device)
        output = self(input_ids=encoded_input.input_ids, attention_mask=encoded_input.attention_mask)
        output: torch.Tensor
        return output.squeeze(0)

#Data loader loads prompt data from json file and translate it using meta-nllb200 and encode it using bert tokenizer.
class MyClassificationNetDataSet(torch.utils.data.Dataset):
    dataset: list[dict[str, typing.Any]]

    def __init__(self, model: MyClassifierNet, file_path: str) -> None:
        super().__init__()
        self.model = model
        self.device = model.device

        with open(file_path) as f:
            self.dataset = json.load(f)
        print("[OK] of loading led control dataset")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> Any:
        item = self.dataset[index]
        prompt = item["prompt"]
        prompt: str
        prompt = self.model.translate(prompt)
        encoded_input = self.model.bert_tokenizer.encode_plus(prompt, add_special_tokens=True, truncation=True, padding="max_length", max_length=32, return_tensors="pt").to(self.device)
        return_Value = ({"input_ids": torch.squeeze(encoded_input.input_ids, 0), "attention_mask": torch.squeeze(encoded_input.attention_mask, 0)}, torch.tensor(item["label"]).to(self.device))
        return return_Value
    


#Train the model
def train_model(model: MyClassifierNet, device: torch.device):
    model.train()
    import matplotlib.pyplot as plt
    plt.ion()
    figure, axis = plt.subplots()
    x, y = [], []

    train_loader = torch.utils.data.DataLoader(MyClassificationNetDataSet(model=model, file_path="./dataset/led_control_trainset.json"), batch_size=63, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(MyClassificationNetDataSet(model=model, file_path="./dataset/led_control_testset.json"), batch_size=63, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    loss_function = torch.nn.CrossEntropyLoss()

    dataset = list(train_loader)

    for i in range(600):
        for data, label in dataset:
            output = model(**data)
            loss = loss_function(output, label)
            loss: torch.Tensor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # for data, label in test_loader:
        #     output = model(**data)
        #     loss = loss_function(output, label)
            
            #visualization the loss.
            x.append(i)
            y.append(loss.item())
            axis.clear()
            axis.scatter(x, y, 0.9)
            axis.set_xlabel(f"epoch: {i}")
            axis.set_ylabel(f"loss: {loss.item():.6f}")
            plt.pause(0.1)
            print(f"Epoch: {i}, Loss: {loss.item()}")

def test_model(model: MyClassifierNet, device: torch.device):
    while True:
        output = model.predict(input("Please enter a prompt>  "))
        print(output)
        index = output.argmax().item()
        print(["Unsupported operation", "LED is ON", "LED is OFF", "Turn on computer", "Turn off computer", "ئادەم تىللىدى", "ئادەمنى ماختىدى"][index])



def main():
    device = torch.device("cuda:0")
    model = MyClassifierNet(output_count=7, device=device).to(device)
    # model.load_state_dict(torch.load("led_controller.pth"))
    train_model(model, device)
    # test_model(model, device)
    
    torch.save(model.state_dict(), "led_controller.pth")


main()
