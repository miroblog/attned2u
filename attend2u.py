import numpy as np
from keras.layers import Concatenate, Lambda, Input, Dense, Conv2D, MaxPooling2D, MaxPooling1D, ZeroPadding2D
from keras.models import Model
import keras.backend as K

N_VOCAB = 60 * 1000 # 40*1000 for post completion
EMB_DIM = 512
MEM_DIM = 1024
IMAGE_RAW_DIM = 2048
D_FREQ_WORDS = 100
N_R5C = 49
CNN_SINGLE = False

image_descriptor = Input(shape=(N_R5C,IMAGE_RAW_DIM), name='image_desc') # (-1, 49, 2048)
user_context = Input(shape=(D_FREQ_WORDS, N_VOCAB), name='user_context') # (-1, 100, 60000)
prev_word = Input(shape=(N_VOCAB,), name='prev_word') # (-1, 60000)


# image memory
im_a_list = []
im_c_list = []

W_im_a = Dense(1024 , activation='relu')
W_im_c = Dense(1024 , activation='relu')

for i in range(N_R5C):
    out = Lambda(lambda x: x[:, i])(image_descriptor) # (-1, 2048)

    im_a_unit = W_im_a(out) # (-1, 1024)
    im_a_list.append(im_a_unit)

    im_c_unit = W_im_c(out) # (-1, 1024)
    im_c_list.append(im_c_unit)

mem_im_a= Lambda(lambda x: K.stack(x, axis=1))(im_a_list) # (-1, 49, 1024)
mem_im_c= Lambda(lambda x: K.stack(x, axis=1))(im_c_list) # (-1, 49, 1024)


# user context memory
W_e_a = Dense(EMB_DIM, activation='linear', use_bias=False) # shared weights <for embedding>
W_e_c = Dense(EMB_DIM, activation='linear', use_bias=False)
W_h = Dense(1024, activation='relu')

us_a_list = []
us_c_list = []

for i in range(D_FREQ_WORDS):
    out = Lambda(lambda x: x[:, i])(user_context) # (-1, 60000)
    # out = Reshape((N_VOCAB,))(out)

    u_j_a = W_e_a(out)  # simple dot product # (-1, 512)
    u_j_c = W_e_c(out)  # compute embedding # (-1, 512)

    us_a_unit = W_h(u_j_a) #(-1, 1024)
    us_c_unit = W_h(u_j_c) #(-1, 1024)

    us_a_list.append(us_a_unit)
    us_c_list.append(us_c_unit)

mem_us_a = Lambda(lambda x: K.stack(x, axis=1))(us_a_list) # (-1, 100, 1024)
mem_us_c = Lambda(lambda  x: K.stack(x, axis=1))(us_c_list) # (-1, 100, 1024)


M_t_a = Concatenate(axis=1)([mem_im_a, mem_us_a]) # (-1, 149, 1024)
M_t_c = Concatenate(axis=1)([mem_im_c, mem_us_c]) # (-1, 149, 1024)
# length of image + user_context memory
mem_length = len(im_a_list) + len(us_a_list) # 149 = 49 + 100

# prediction
W_e_b = Dense(EMB_DIM, activation='linear', use_bias=False)
x_t = W_e_b(prev_word) # query to embedding (-1, 512)
q_t = Dense(1024, activation='relu')(x_t) # query_embedding to memory (-1, 1024)

# compute attention with query, M_t_a
# (-1, 149) = (-1,149, 1024) batch_dot (-1, 1024)
# then softmax to compute attention
p_t = Lambda(lambda x: K.softmax(K.batch_dot(x, q_t)))(M_t_a)
# element-wise multiplication, (-1, 149, 1024) * (-1, 149, 1)
p_t = Lambda(lambda x : K.reshape(x, shape=(-1, mem_length, 1)))(p_t)
M_o_t = Lambda(lambda x: x[0] * x[1])([M_t_c, p_t]) #M_t_c * p_t

# single layer cnn
# compute convolution
def conv_memory(mem_length, memory):
    # expand last dimension for conv2d (-1, 49, 1024) -> (-1, 49, 1024, 1)
    memory_3d = Lambda(lambda  x: K.expand_dims(x, axis=-1))(memory)
    kernel_sizes = [3, 4, 5]
    convs = []
    for k in kernel_sizes:
        # (-1, 47 , 1, 300 )
        conv = Conv2D(filters=300, kernel_size=(k, 1024), padding='valid', activation='relu')(memory_3d)  # no zero padding
        # (-1, 1, 1, 300)
        c_im_t = MaxPooling2D(pool_size=(mem_length - k, 1))(conv)
        convs.append(c_im_t)
    return Concatenate()(convs)

# refer to ... https://arxiv.org/pdf/1612.08083.pdf
def apply_glu(prev_out): # prev_out (-1, 49, 1024)

    # expand last dimension for conv2d (-1, 49, 1024) -> (-1, 49, 1024, 1)
    memory_3d = Lambda(lambda  x: K.expand_dims(x, axis=-1))(prev_out)
    # padding to reserve dimension
    # (0,0),(0,2) (top, bottom), (left, right) (-1, 49, 1024, 1) -> (-1, 51, 1024, 1)
    memory_3d = ZeroPadding2D(padding=((0,2),(0,0)))(memory_3d)

    k = 3 # kernel size
    conv1 = Conv2D(filters=1024, kernel_size=(k, 1024), strides=1, padding='valid', activation='linear')(memory_3d)  # to attain (-1, 49,1,1024)
    conv1 = Lambda(lambda x: K.squeeze(x, axis=2))(conv1) # (-1, 49, 1024)

    sigmoid_conv2 = Conv2D(filters=1024, kernel_size=(k, 1024), padding='valid', activation='sigmoid')(memory_3d)  # this is the gate
    sigmoid_conv2 = Lambda(lambda x: K.squeeze(x, axis=2))(sigmoid_conv2)  # (-1, 49, 1024)

    h_t = Lambda(lambda x: x[0] * x[1])([conv1, sigmoid_conv2]) # (-1, 49, 1024) * (-1, 49, 1024)
    h_t = Lambda(lambda x: x[0] + x[1])([h_t, prev_out]) # (-1, 49, 1024)
    return h_t

def construct_mul_cnn(n_layers, memory, mem_length):
    conv_output = memory
    for _ in range(n_layers):
        conv_output = apply_glu(conv_output)
    # maxpool at final layer
    conv_output = MaxPooling1D(pool_size=(mem_length))(conv_output) # (-1, 1024)
    return conv_output

# again..., partition memory
# image [:49] user context [49:]
m_im_o = Lambda(lambda x: x[:, :len(im_a_list)])(M_o_t)
m_us_o = Lambda(lambda x: x[:, len(im_a_list):])(M_o_t)

if(CNN_SINGLE):
    c_im_t = conv_memory(len(im_a_list), m_im_o)
    c_us_t = conv_memory(len(us_a_list), m_us_o)
    # (-1, 1, 1, 1800)
    c_t = Concatenate()([c_im_t, c_us_t]) # 2*3*300 = 1800
    # (-1, 1800)
    c_t = Lambda(lambda x: K.reshape(x, shape=(-1, 1800)))(c_t)
else: # CNN Multiple
    c_im_t = construct_mul_cnn(3, m_im_o, len(im_a_list)) # (-1, 1024)
    c_us_t = construct_mul_cnn(3, m_us_o, len(us_a_list)) # (-1, 1024)
    c_t = Concatenate()([c_im_t, c_us_t]) # (-1, 2048)

h_t = Dense(1800, activation='relu')(c_t)
s_t = Dense(N_VOCAB, activation='softmax')(h_t)

model = Model([image_descriptor, user_context], s_t)
model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)