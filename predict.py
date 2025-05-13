import json
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import time
import os

# 读取类别标签
with open('classes.json', 'r', encoding='utf-8') as f:
    classes = json.load(f)

# 定义模型结构（和训练时完全一致）
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # 自动适配不同图片尺寸
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 预测图片函数
def predict_image(image_path, model_path, classes, top_k=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = CNN(num_classes=len(classes)).to(device)
    # weights_only=True 会限制 PyTorch 只加载模型的权重，而不会加载额外的对象（如模型结构之外的其他对象），这可以提高安全性。
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 读取图片，自动转 RGB 防止灰度/透明通道报错
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {"error": f"图片读取失败: {str(e)}"}

    image = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度

    # 推断并计时
    start_time = time.time()
    with torch.no_grad():
        output = model(image)
        probabilities = nn.functional.softmax(output, dim=1)
        confidences, predicted = torch.topk(probabilities, k=top_k)

    end_time = time.time()

    predicted_classes = [classes[i] for i in predicted[0].tolist()]
    confidence_scores = confidences[0].tolist()

    return {
        "predicted_class": predicted_classes[0],
        "confidence": confidence_scores[0],
        "top_k_predictions": list(zip(predicted_classes, confidence_scores)),
        "predict_time_sec": round(end_time - start_time, 4)
    }

# API封装函数
def predict_image_api(image_path):
    model_path = './animal_classification_model.pth'
    if not os.path.exists(image_path):
        return {"error": "图片路径不存在！"}
    return predict_image(image_path, model_path, classes)

# 本地测试用
if __name__ == "__main__":
    image_path = './test.jpg'
    result = predict_image_api(image_path)

    if "error" in result:
        print(result["error"])
    else:
        print(f'预测类别: {result["predicted_class"]}')
        print(f'置信度: {result["confidence"]:.4f}')
        print(f'推断耗时: {result["predict_time_sec"]} 秒')
        print('Top-K 预测:')
        for cls, score in result['top_k_predictions']:
            print(f'  类别: {cls}, 置信度: {score:.4f}')
