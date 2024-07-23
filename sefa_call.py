import os
from tqdm import tqdm
import numpy as np

import torch

from models import parse_gan_type
from utils import to_tensor
from utils import postprocess
from utils import load_generator
from utils import factorize_weight
from utils import HtmlPageVisualizer

gpu_id = '0'
save_dir = 'results'
model_name =  'stylegan2ada_brecahad'
layer_idx = "all"
seed = 0
num_samples = 5
trunc_psi = 0.7
trunc_layers=8
step = 11
start_distance = -3.0
end_distance = 3.0
num_semantics = 5
viz_size = 256
label_size=None



os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
os.makedirs(save_dir, exist_ok=True)

# Factorize weights.
generator = load_generator(model_name)
gan_type = parse_gan_type(generator)
layers, boundaries, values = factorize_weight(generator, layer_idx)

# Set random seed.
np.random.seed(seed)
torch.manual_seed(seed)

# Prepare codes.
codes = torch.randn(num_samples, generator.z_space_dim).cuda()
# create array of labels and repeat them for num_samples
if label_size is None:
    labels = None
else:
    label = [1, 0, 1, 0, 0, 0]
    labels = torch.tensor(label, dtype=torch.float32).repeat(num_samples, 1).cuda()

if gan_type == 'pggan':
    codes = generator.layer0.pixel_norm(codes)
elif gan_type in ['stylegan', 'stylegan2', 'stylegan2ada']:
    codes = generator.mapping(codes, labels)['w']
    codes = generator.truncation(codes,
                                 trunc_psi=trunc_psi,
                                 trunc_layers=trunc_layers)
codes = codes.detach().cpu().numpy()

# Generate visualization pages.
distances = np.linspace(start_distance,end_distance, step)  
num_sam = num_samples
num_sem = num_semantics
vizer_1 = HtmlPageVisualizer(num_rows=num_sem * (num_sam + 1),
                                num_cols=step + 1,
                                viz_size=viz_size)
vizer_2 = HtmlPageVisualizer(num_rows=num_sam * (num_sem + 1),
                                num_cols=step + 1,
                                viz_size=viz_size)

headers = [''] + [f'Distance {d:.2f}' for d in distances]
vizer_1.set_headers(headers)
vizer_2.set_headers(headers)
for sem_id in range(num_sem):
    value = values[sem_id]
    vizer_1.set_cell(sem_id * (num_sam + 1), 0,
                        text=f'Semantic {sem_id:03d}<br>({value:.3f})',
                        highlight=True)
    for sam_id in range(num_sam):
        vizer_1.set_cell(sem_id * (num_sam + 1) + sam_id + 1, 0,
                            text=f'Sample {sam_id:03d}')
for sam_id in range(num_sam):
    vizer_2.set_cell(sam_id * (num_sem + 1), 0,
                        text=f'Sample {sam_id:03d}',
                        highlight=True)
    for sem_id in range(num_sem):
        value = values[sem_id]
        vizer_2.set_cell(sam_id * (num_sem + 1) + sem_id + 1, 0,
                            text=f'Semantic {sem_id:03d}<br>({value:.3f})')

for sam_id in tqdm(range(num_sam), desc='Sample ', leave=False):
    code = codes[sam_id:sam_id + 1]
    for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
        boundary = boundaries[sem_id:sem_id + 1]
        for col_id, d in enumerate(distances, start=1):
            temp_code = code.copy()
            if gan_type == 'pggan':
                temp_code += boundary * d
                image = generator(to_tensor(temp_code))['image']
            elif gan_type in ['stylegan', 'stylegan2', 'stylegan2ada']:
                temp_code[:, layers, :] += boundary * d
                image = generator.synthesis(to_tensor(temp_code))['image']
            image = postprocess(image)[0]
            vizer_1.set_cell(sem_id * (num_sam + 1) + sam_id + 1, col_id,
                                image=image)
            vizer_2.set_cell(sam_id * (num_sem + 1) + sem_id + 1, col_id,
                                image=image)

prefix = (f'{model_name}_'
            f'N{num_sam}_K{num_sem}_L{layer_idx}_seed{seed}')
vizer_1.save(os.path.join(save_dir, f'{prefix}_sample_first.html'))
vizer_2.save(os.path.join(save_dir, f'{prefix}_semantic_first.html'))
