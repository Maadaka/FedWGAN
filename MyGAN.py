# import torch.nn as nn
#
#
# class Generator(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Generator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, output_size),
#             nn.Tanh()  # Tanh helps to keep the output values between -1 and 1
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class Discriminator(nn.Module):
#     def __init__(self, input_size):
#         super(Discriminator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.net(x)


# 这是WGAN
import torch.nn as nn

# 生成器
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_size),
            nn.Tanh()  # 保留 Tanh 激活函数
        )

    def forward(self, x):
        return self.net(x)

# Critic
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # 去掉 Sigmoid 激活函数
        )

    def forward(self, x):
        return self.net(x)