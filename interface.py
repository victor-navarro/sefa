# python 3.7
"""Demo."""

import numpy as np
import torch
import streamlit as st

from models import parse_gan_type
from utils import to_tensor
from utils import postprocess
from utils import load_generator
from utils import factorize_weight

if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'code_idx' not in st.session_state:
    st.session_state.code_idx = None
if 'codes' not in st.session_state:
    st.session_state.codes = None
if 'cat' not in st.session_state:
    st.session_state.cat = None


@st.cache_data(show_spinner=False)
def get_model(model_name):
    """Gets model by name."""
    return load_generator(model_name)


@st.cache_data(show_spinner=False)
def factorize_model(_model, layer_idx):
    """Factorizes semantics from target layers of the given model."""
    return factorize_weight(_model, layer_idx)


def sample(model, gan_type, model_name, cat, num=1):
    """Samples latent codes."""
    codes = torch.randn(num, model.z_space_dim).cuda()
    if gan_type == 'pggan':
        codes = model.layer0.pixel_norm(codes)
    elif gan_type == 'stylegan':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.7,
                                 trunc_layers=8)
    elif gan_type == 'stylegan2':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.5,
                                 trunc_layers=18)
    elif gan_type == 'stylegan2ada':
        labels = get_labels(model_name, cat, n=num)
        codes = model.mapping(codes, labels)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.5,
                                 trunc_layers=8)
    codes = codes.detach().cpu().numpy()
    return codes


@st.cache_data(show_spinner=False)
def synthesize(_model, gan_type, code):
    """Synthesizes an image with the give code."""
    if gan_type == 'pggan':
        image = _model(to_tensor(code))['image']
    elif gan_type in ['stylegan', 'stylegan2', 'stylegan2ada']:
        image = _model.synthesis(to_tensor(code))['image']
    image = postprocess(image)[0]
    return image


def main():
    """Main function (loop for StreamLit)."""
    st.title('Closed-Form Factorization of Latent Semantics in GANs')
    st.sidebar.title('Options')
    reset = st.sidebar.button('Reset')

    model_name = st.sidebar.selectbox(
        'Model to Interpret',
        ['stylegan2ada_6Mhist', 'stylegan2ada_brecahad',
         'stylegan2_ffhq', 
         'stylegan_animeface512', 'stylegan_car512', 
         'stylegan_cat256', 'pggan_celebahq1024'])

    cat = get_catselection(model_name)

    model = get_model(model_name)
    gan_type = parse_gan_type(model)
    layer_idx = st.sidebar.selectbox(
        'Layers to Interpret',
        ['all', '0-1', '2-5', '6-13'])
    layers, boundaries, eigen_values = factorize_model(model, layer_idx)

    num_semantics = st.sidebar.number_input(
        'Number of semantics', value=10, min_value=0, max_value=None, step=1)
    steps = {sem_idx: 0 for sem_idx in range(num_semantics)}
    if gan_type == 'pggan':
        max_step = 5.0
    elif gan_type in ['stylegan']:
        max_step = 2.0
    elif gan_type in ['stylegan2']:
        max_step = 15.0
    elif gan_type in ['stylegan2ada']:
        max_step = 15.0
    for sem_idx in steps:
        eigen_value = eigen_values[sem_idx]
        steps[sem_idx] = st.sidebar.slider(
            f'Semantic {sem_idx:03d} (eigen value: {eigen_value:.3f})',
            value=0.0,
            min_value=-max_step,
            max_value=max_step,
            step=0.04 * max_step)

    image_placeholder = st.empty()
    button_placeholder = st.empty()

    if cat != st.session_state.cat:
        st.session_state.codes = sample(model, gan_type, model_name, cat)
    else:
        if model_name != 'stylegan2ada_6Mhist':
            cat = None
        
    st.session_state.cat = cat

    try:
        base_codes = np.load(f'C:/gancat-local/sefa/latent_codes/{model_name}_latents.npy')
    except FileNotFoundError:
        base_codes = sample(model, gan_type, model_name, cat)
        
    if st.session_state.model_name != model_name:
        st.session_state.model_name = model_name
        st.session_state.code_idx = 0
        st.session_state.codes = base_codes[0:1]

    if button_placeholder.button('Random', key=0):
        st.session_state.code_idx += 1
        if st.session_state.code_idx < base_codes.shape[0]:
            st.session_state.codes = base_codes[st.session_state.code_idx][np.newaxis]
        else:
            st.session_state.codes = sample(model, gan_type, model_name, cat)



    code = st.session_state.codes.copy()
    for sem_idx, step in steps.items():
        if gan_type == 'pggan':
            code += boundaries[sem_idx:sem_idx + 1] * step
        elif gan_type in ['stylegan', 'stylegan2', 'stylegan2ada']:
            code[:, layers, :] += boundaries[sem_idx:sem_idx + 1] * step
    image = synthesize(model, gan_type, code)
    image_placeholder.image(image / 255.0)


def get_labels(model_name, category, n=1):
    if model_name == 'stylegan2ada_6Mhist':
        label_dict = {'B40': [1, 0, 1, 0, 0, 0],
                    'B100': [1, 0, 0, 1, 0, 0],
                    'B200': [1, 0, 0, 0, 1, 0],
                    'B400': [1, 0, 0, 0, 0, 1],
                    'M40': [0, 1, 1, 0, 0, 0],
                    'M100': [0, 1, 0, 1, 0, 0],
                    'M200': [0, 1, 0, 0, 1, 0],
                    'M400': [0, 1, 0, 0, 0, 1]}
        return torch.tensor(label_dict[category], dtype=torch.float32).repeat(n, 1).cuda()
    return None

def get_catselection(model_name):
    if model_name == 'stylegan2ada_6Mhist':
        return st.sidebar.selectbox(
            'Category',
            ['B40', 'B100', 'B200', 'B400', 'M40', 'M100', 'M200', 'M400'])
    return None

if __name__ == '__main__':
    main()
