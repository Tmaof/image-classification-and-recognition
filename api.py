from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from predict import predict_image_api

app = Flask(__name__)
# 允许中文输出，解决 接口返回的中文字符中为：\u732b\u7c7b-\u6a58\u732b格式
app.config['JSON_AS_ASCII'] = False

# 设置允许的文件扩展名 (api.py)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# 文件部分的键名
FILE_PART_KEY = 'file'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def upload_file():
    if FILE_PART_KEY not in request.files:
        return jsonify({"error": f'formdata 中图片的key需要为 {FILE_PART_KEY}'}), 400
    file = request.files[FILE_PART_KEY] #获取上传的图像文件。
    
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        # 调用预测函数
        result = predict_image_api(filepath)
        
        # 删除临时文件
        os.remove(filepath)
        
        # 确保返回的 JSON 数据包含直接可读的中文字符
        response = jsonify({
            "data": result,
            "message": "请求成功",
            "success": True,
        })
        # response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    else:
        return jsonify({"message": "图片格式不允许", "success": False,}), 400

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    port = 5000
    print("图片分类识别api运行在端口：" ,port)
    app.run(port=port,debug=True)