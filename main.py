import torch.autograd
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

if not os.path.exists('./img') :
    os.mkdir('./img')
def to_img(x):
    out=0.5*(x+1)
    out=out.clamp(0,1)
    out=out.view(-1,1,28,28)
    return out
batch_size=128
epoch=100
z_dimension=100
img_transform=transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,),(0.5,))]) #串联定义图像变换

#数据集下载，取出训练集
minst=datasets.MNIST(root='./data',train=True,transform=img_transform,download=True)
dataloader=DataLoader(dataset=minst,batch_size=batch_size,shuffle=True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disnet=nn.Sequential(
            nn.Linear(784,256),
            nn.LeakyReLU(0,2),
            nn.Linear(256,256),
            nn.LeakyReLU(0,2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,input):
        input=self.disnet(input)
        return input

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.genenet=nn.Sequential(
            nn.Linear(100,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,784),
            nn.Tanh()
        )
    def forward(self,input):
        input=self.genenet(input)
        return input
#测试commit功能能够实现
generator=Generator()
discriminator=Discriminator()
if torch.cuda.is_available() :
    generator=generator.cuda()
    discriminator=discriminator.cuda()
g_optimizer=torch.optim.Adam(generator.parameters(),lr=0.0003)
d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=0.0003)
loss=nn.BCELoss() #损失
#训练生成器和判断器
for epo in range(epoch):
    for index,(img,_) in enumerate(dataloader): #只拿训练图片，不拿标签
        num_img=img.size(0)  #获取一个batch的图片数量
        img=img.view(num_img,-1) #将img向量展平 全连接  相当于img=torch.flatten(img,1)
        real_img=Variable(img).cuda();
        real_label=Variable(torch.ones(num_img)).cuda(); #128个变量
        fake_label=Variable(torch.zeros(num_img)).cuda();

        #判别器训练
        real_out=discriminator(real_img)
        real_out=real_out.squeeze()  #squeeze()将维度为1的那一维去掉
        real_loss=loss(real_out,real_label)
        real_score=real_out

        z=Variable(torch.randn(num_img,z_dimension)).cuda() #输出num_img×zdimension
        fake_img=generator(z).detach()  #反向传播generator参数不更新
        fake_out=discriminator(fake_img).squeeze()
        fake_loss=loss(fake_out,fake_label)
        fake_score=real_out

        d_loss=real_loss+fake_loss
        d_optimizer.zero_grad() #反向传播前先将梯度归零
        d_loss.backward() #误差反向传播计算梯度
        d_optimizer.step()  #parameter参数更新

        #--------------------------生成器
        z=Variable(torch.randn(num_img,z_dimension)).cuda()
        fake_img2=generator(z)
        fake_out2=discriminator(fake_img2)#--------注意这里不需要detach
        fake_out2=fake_out2.squeeze()
        fake_score2=fake_out2
        g_loss=loss(fake_out2,real_label)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        # print(fake_score2)
        if  (index+1)%100==0:
            print('epoch:{}/{},d_loss:{:.6f},g_loss:{:.6f},fake_score2:{:.6f}'.format(
                epo,epoch,d_loss.item(),g_loss.item(),fake_score2.data.mean()
            ))
        fake_images=to_img(fake_img2.cpu().data)
        save_image(fake_images,'./img/fake_img{}.png'.format(epo+1))



torch.save(generator.state_dict(),'./generator.pth')
torch.save(discriminator.state_dict(),'./discriminator.pth')

