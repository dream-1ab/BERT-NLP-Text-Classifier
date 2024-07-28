

# import torch
# import helper

# device = torch.device("cuda:0")

# class MyBinaryCalculator(torch.nn.Module):
#     layers: list[torch.nn.Linear]
#     _activation_function = torch.nn.ReLU()
#     def __init__(self):
#         super(MyBinaryCalculator, self).__init__()
#         self.layers = []
#         self.layers.append(torch.nn.Linear(16, 128))
#         self.layers.append(torch.nn.Linear(128, 128))
#         self.layers.append(torch.nn.Linear(128, 64))
#         self.layers.append(torch.nn.Linear(64, 8))
#         self.fc0 = self.layers[0]
#         self.fc1 = self.layers[1]
#         self.fc2 = self.layers[2]
#         self.fc3 = self.layers[3]

#     def forward(self, x: torch.Tensor):
#         for index, layer in enumerate(self.layers):
#             x = layer(x)
#             if index + 1 != len(self.layers):
#                 x = self._activation_function(x)
#         return x

# def train(model: torch.nn.Module, dataset: list[tuple[torch.Tensor, torch.Tensor]]):
#     loss_function = torch.nn.MSELoss()
#     params = model.parameters()
#     optimizer = torch.optim.SGD(params, 0.001)
#     model.train()
#     for i in range(1000):
#         loss: torch.Tensor
#         batch = [[], []]
#         for index, (x, y) in enumerate(dataset):
#             batch[0].append(x)
#             batch[1].append(y)

#         optimizer.zero_grad()
#         x = torch.stack(batch[0]).to(device)
#         y = torch.stack(batch[1]).to(device)
#         output = model(x)
#         loss = loss_function(output, y)
#         loss.backward()
#         optimizer.step()
#         print(f"{i} loss: {loss.item()}")
#     model.eval()

# def prepare_dataset() -> list[tuple[torch.Tensor, torch.Tensor]]:
    
            
#     dataset = []
#     for i in range(120):
#         for x in range(120):
#             a = helper.number_to_binary_tensor(i)
#             b = helper.number_to_binary_tensor(x)

#             dataset.append((torch.concat((a, b)), helper.number_to_binary_tensor(i + x)))
#     return dataset


# nn = MyBinaryCalculator().to(device)
# # nn.load_state_dict(torch.load("nn.pth"))
# dataset = prepare_dataset()
# train(model=nn, dataset=dataset)

# input = torch.concat((helper.number_to_binary_tensor(20), helper.number_to_binary_tensor(20))).to(device)
# output = nn(input)
# output = helper.binary_tensor_to_number(output)
# print(output)
# torch.save(nn.state_dict(), "nn.pth")



# import torch

# device = torch.device("cuda:0")

# class AdderNN(torch.nn.Module):
#     def __init__(self) -> None:
#         super(AdderNN, self).__init__()
#         from torch.nn import Linear, Sigmoid
#         self.fc0 = Linear(2, 512)
#         self.fc1 = Linear(512, 2048)
#         self.fc2 = Linear(2048, 1024)
#         self.fc3 = Linear(1024, 512)
#         self.fc4 = Linear(512, 1)
#         self.activator = Sigmoid()
    
#     def forward(self, x):
#         x = self.activator(self.fc0(x))
#         x = self.activator(self.fc1(x))
#         x = self.activator(self.fc2(x))
#         x = self.activator(self.fc3(x))
#         x = self.fc4(x)
#         return x


# def make_dataset(count: int):
#     xs = []
#     ys = []
#     for i in range(count):
#         x = torch.randint(0, 100, (1,)).to(device).float()
#         y = torch.randint(0, 100, (1,)).to(device).float()
#         row = torch.concat([x, y])
#         xs.append(row)
#         result = x + y
#         ys.append(result)
        
#     return (torch.stack(xs).to(device), torch.stack(ys).to(device))

# def train(model: torch.nn.Module, dataset: tuple[torch.Tensor, torch.Tensor]):
#     model.train()
#     loss_function = torch.nn.MSELoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#     for i in range(1000):
#         optimizer.zero_grad()
#         loss: torch.Tensor

#         x, y = dataset
#         output = model(x)
#         loss = loss_function(output, y)
#         loss.backward()
#         optimizer.step()
#         print(f"{i}: {loss.item()}")
#     model.eval()


# nn = AdderNN().to(device)
# nn.load_state_dict(torch.load("nn.pth"))

# # dataset = make_dataset(100000)
# # train(nn, dataset)
# # torch.save(nn.state_dict(), "nn.pth")

# output = nn(torch.tensor([20, 30]).to(device).float())
# print(output.item())
# print("Done.")


import torch

device = torch.device("cuda:0")

class MyAdderNet(torch.nn.Module):
    def __init__(self) -> None:
        super(MyAdderNet, self).__init__()
        from torch.nn import Linear, Sigmoid, ReLU, Tanh
        self.fc0 = Linear(4, 64)
        self.fc1 = Linear(64, 64)
        # self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 2)
        self.activation_function = ReLU()
    
    def forward(self, input):
        input = self.fc0(input)
        input = self.activation_function(input)
        input = self.fc1(input)
        # input = self.activation_function(input)
        # input = self.fc2(input)
        # input = self.activation_function(input)
        input = self.fc3(input)
        return input

def train(model: MyAdderNet, sample_count=20000, epoch=1000):
    def make_sample_data(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        input = torch.randn([sample_count, 4]).to(device)
        result = torch.concat((input[:, 0:1] + input[:, 1:2], input[:, 2:3] - input[:, 3:4]), dim=1)
        return (input, result)
    
    data_input, data_output = make_sample_data(sample_count)
    test_input, test_output = make_sample_data(1000)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


    x: list[int] = []
    y: list[float] = []
    
    import matplotlib.pyplot as plt
    plt.ion()
    figure, axis = plt.subplots()
    
    def refresh_plot(x_label: str, y_label: str):
        axis.clear()
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.scatter(x, y, 0.8)
        figure.canvas.draw()
        plt.pause(0.01)


    for i in range(epoch):
        output = model(data_input)
        loss = loss_function(output, data_output)
        loss: torch.Tensor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i > 100 and i % 100 == 0:
            output = model(test_input)
            loss = loss_function(output, test_output)

            x.append(i)
            y.append(loss.item())
            refresh_plot(x_label=f"epoch ({i})", y_label=f"loss ({(loss.item()):.6f}")
            print(f"epoch: {i} loss: {loss.item()}")

model = MyAdderNet().to(device)
model.load_state_dict(torch.load("my_Adder.pth"))

train(model, sample_count=50_000, epoch=500000)

torch.save(model.state_dict(), "my_Adder.pth")

input = torch.tensor([0.5, 0.1, 0.2, 0.8]).to(device)
output = model(input)
print(output)

def export_model(model: MyAdderNet):
    torch.onnx.export(model=model, args=torch.randn([1, 4]).to(device), f="my_Adder.onnx")

def test_model(model: MyAdderNet):
    import matplotlib.pyplot as plt
    data = []
    for i in range(1000):
        i /= 1000
        data.append([0, i, 0.5, i])
    input = torch.tensor(data).to(device)
    output = model(input)
    output: torch.Tensor

    x = []
    y = []
    for i, item in enumerate(output.tolist()):
        x.append(i / 1000)
        x.append(i / 1000)
        y.append(item[0])
        y.append(item[1])

    plt.scatter(x, y, 0.5)
    plt.show()

test_model(model)

export_model(model)