import os
from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import faiss

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
device = torch.device('cpu')
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def get_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image).squeeze().cpu().numpy()
    return embedding

# Load image database
filenames = []
embeddings = []
for fname in os.listdir(IMAGE_FOLDER):
    if fname.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        path = os.path.join(IMAGE_FOLDER, fname)
        try:
            emb = get_embedding(path)
            embeddings.append(emb)
            filenames.append(fname)
        except:
            print(f"Error processing {fname}")

embedding_matrix = np.array(embeddings).astype('float32')
faiss_index = faiss.IndexFlatL2(embedding_matrix.shape[1])
faiss_index.add(embedding_matrix)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_image']
        if file and file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            query_vec = get_embedding(filepath).astype('float32').reshape(1, -1)
            D, I = faiss_index.search(query_vec, k=5)
            matches = [(filenames[idx], float(D[0][i])) for i, idx in enumerate(I[0])]

            return render_template('index.html', uploaded_image=file.filename, matches=matches)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
