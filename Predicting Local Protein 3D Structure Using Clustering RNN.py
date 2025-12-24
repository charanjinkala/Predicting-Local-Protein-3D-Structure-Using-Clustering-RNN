#!pip install torch numpy biopython py3Dmol requests 
 
from io import StringIO 
import torch 
import torch.nn as nn 
import numpy as np 
from Bio import SeqIO 
from urllib.request import urlopen 
import requests 
import py3Dmol 
from IPython.display import HTML 
 
# 🔹 Fetch sequence from UniProt 
def fetch_uniprot_sequence(uniprot_id): 
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta" 
    with urlopen(url) as response: 
        fasta_text = response.read().decode("utf-8") 
        fasta_io = StringIO(fasta_text) 
        record = SeqIO.read(fasta_io, "fasta") 
    return str(record.seq) 
 
# 🔹 Segment and mock HSSP profile 
def generate_segments(sequence, window_size=15): 
    return [sequence[i:i+window_size] for i in range(len(sequence) - window_size + 1)] 
 
def mock_hssp_profile(segment): 
    return np.random.rand(len(segment), 20) 
 
# 🔹 CRNN model 
class CRNN(nn.Module): 
    def __init__(self, input_size=20, hidden_size=64, output_size=3): 
        super(CRNN, self).__init__() 
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True) 
        self.fc = nn.Linear(hidden_size, output_size) 
 
    def forward(self, x): 
        out, _ = self.rnn(x) 
        return self.fc(out[:, -1, :]) 
 
# 🔹 Predict angles and convert to coordinates 
def predict_angles(model, profiles): 
    predicted_angles = [] 
    for profile in profiles: 
        input_tensor = torch.tensor(profile, dtype=torch.float32).unsqueeze(0) 
24  
        angles = model(input_tensor).detach().numpy()[0] 
        predicted_angles.append(angles) 
    return predicted_angles 
 
def angles_to_coordinates(angles): 
    coords = [] 
    x, y, z = 0.0, 0.0, 0.0 
    for phi, psi, omega in angles: 
        x += np.cos(phi) 
        y += np.sin(psi) 
        z += np.sin(omega) 
        coords.append((x, y, z)) 
    return coords 
 
# 🔹 Save predicted structure to PDB 
def save_pdb(coords, filename="predicted_structure.pdb"): 
    with open(filename, "w") as f: 
        for i, (x, y, z) in enumerate(coords): 
            f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  
1.00  0.00           C\n") 
 
# 🔹 Download AlphaFold structure 
def download_alphafold_pdb(uniprot_id, filename="alphafold_structure.pdb"): 
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb" 
    response = requests.get(url) 
    if response.status_code == 200: 
        with open(filename, "w") as f: 
            f.write(response.text) 
        print("✅ AlphaFold structure downloaded.") 
    else: 
        print("❌ Failed to download AlphaFold structure.") 
 
# 🔹 Visualize any PDB file 
def show_structure(pdb_file, title="Protein Structure"): 
    with open(pdb_file, "r") as f: 
        pdb_data = f.read() 
    view = py3Dmol.view(width=800, height=500) 
    view.addModel(pdb_data, "pdb") 
    view.setStyle({"cartoon": {"color": "spectrum"}}) 
    view.zoomTo() 
    return HTML(f"<h3>{title}</h3>" + view._make_html()) 
 
# 🔧 Run everything 
uniprot_id = "P69905"  # Hemoglobin subunit alpha 
 
# Step 1: Predict structure using CRNN 
sequence = fetch_uniprot_sequence(uniprot_id) 
segments = generate_segments(sequence) 
profiles = [mock_hssp_profile(seg) for seg in segments] 
model = CRNN() 
predicted_angles = predict_angles(model, profiles) 
coords = angles_to_coordinates(predicted_angles) 
save_pdb(coords, "predicted_structure.pdb") 
# Step 2: Download AlphaFold structure 
download_alphafold_pdb(uniprot_id) 
# Step 3: Visualize both structures 
show_structure("predicted_structure.pdb", "🧪 CRNN Predicted Structure") 
show_structure("alphafold_structure.pdb", "🧪 AlphaFold Structure")
