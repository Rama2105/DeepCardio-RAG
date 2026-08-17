import json
import os
import io

file_path = os.path.join('d:\\\\', 'personal', 'personal', 'sru', 'RAG', 'DrSSN Sir Final Work', 'sssss', 'DeepCardio_RAG_Google_Colab.ipynb')

with io.open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell.get('metadata', {}).get('id') == 'mount_drive':
        cell['source'] = [
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n",
            "\n",
            "# FIX: Automatically Extract ZIP file if it's there\n",
            "import os\n",
            "import zipfile\n",
            "\n",
            "zip_path = '/content/drive/MyDrive/sssss.zip'\n",
            "extract_path = '/content/drive/MyDrive/sssss'\n",
            "\n",
            "if os.path.exists(zip_path) and not os.path.exists(extract_path):\n",
            "    print(f'Extracting {zip_path}...')\n",
            "    with zipfile.ZipFile(zip_path, 'r') as zip_ref:\n",
            "        zip_ref.extractall('/content/drive/MyDrive/')\n",
            "    print('Extraction complete.')\n",
            "\n",
            "if not os.path.exists(extract_path):\n",
            "    print(f'Path {extract_path} does not exist. Please check your Google Drive.')\n",
            "else:\n",
            "    os.chdir(extract_path)\n",
            "    print('Current working directory:', os.getcwd())\n"
        ]
        break

with io.open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook updated successfully.")
