import torchvision

from src.work import testloader, classes
from src.work2 import imshow
from src.work3 import net

if __name__ == '__main__':
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    outputs = net(images)