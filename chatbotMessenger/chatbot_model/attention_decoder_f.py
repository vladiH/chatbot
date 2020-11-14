# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import regularizers, constraints, initializers, activations
from keras.layers import Recurrent,Layer
from keras.engine import InputSpec
from keras.utils import plot_model
from keras.layers import Dropout
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)
def time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x


# In[3]:


class attention_LSTM(Recurrent):
    '''
    # References
    -[name paper](url)
    '''
    def __init__(self, 
                 units, #units in decoder 
                 steps, # steps for output
                 output_dim, # dimension of output
                 atten_units, #attencion units in dense layer
                 gmax, #sub hidden units maxout
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_probabilities=False,
                 **kwargs):
        
        self.units = units
        self.steps = steps
        self.output_dim = output_dim
        self.atten_units = atten_units
        self.activation = activations.get(activation)
        self.gmax = gmax
        self.recurrent_activation = activations.get(recurrent_activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        
        self.return_probabilities = return_probabilities
        
        """if self.dropout or self.recurrent_dropout:
            self.uses_learning_phase = True"""
        super(attention_LSTM, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        if self.return_sequences:
            if self.return_probabilities:
                output_shape = (input_shape[0][0], self.time_step_e, 1)
            else:
                output_shape = (input_shape[1][0], self.steps, self.output_dim)
        else:
            output_shape = (input_shape[0][0], self.output_dim)

        if self.return_state:
            state_shape = [(input_shape[0][0], dim) for dim in self.states]
            return [output_shape] + state_shape
        else:
            return output_shape
    
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.batch_size, self.time_step, self.input_dim = input_shape[1]
        self.batch_size_e, self.time_step_e, self.input_dim_e = input_shape[0]
        
    
        """
            Matrices for cx (context) gate
        """
        self.C_cx = self.add_weight(shape=(self.atten_units,), name='C_cx',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        self.W_cx = self.add_weight(shape=(self.input_dim_e + self.units, self.atten_units),name='W_cx',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        self.b_cx = self.add_weight(shape=(self.atten_units,), name='b_cx',
                            initializer=self.bias_initializer,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint)
        """
            Matrices for i (input) gate
        """
        self.V_i = self.add_weight(shape=(self.input_dim,self.units),name='V_i',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        self.W_i = self.add_weight(shape=(self.input_dim_e, self.units),name='W_i',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        self.U_i = self.add_weight(shape=(self.units, self.units),name='U_i',
                                   initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)
        self.b_i = self.add_weight(shape=(self.units,), name='b_i',
                          initializer=self.bias_initializer,
                          regularizer=self.bias_regularizer,
                          constraint=self.bias_constraint)
        
        """
            Matrices for f (forget) gate
        """
        self.V_f = self.add_weight(shape=(self.input_dim,self.units),name='V_f',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        self.W_f = self.add_weight(shape=(self.input_dim_e, self.units),name='W_f',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        self.U_f = self.add_weight(shape=(self.units, self.units), name='U_f',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)
        self.b_f = self.add_weight(shape=(self.units,), name='b_f',
                          initializer=self.bias_initializer,
                          regularizer=self.bias_regularizer,
                          constraint=self.bias_constraint)
        
        """
            Matrices for o (output) gate
        """
        self.V_o = self.add_weight(shape=(self.input_dim,self.units),name='V_o',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        self.W_o = self.add_weight(shape=(self.input_dim_e, self.units),name='W_o',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        self.U_o = self.add_weight(shape=(self.units, self.units), name='U_o',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)
        self.b_o = self.add_weight(shape=(self.units,), name='b_o',
                          initializer=self.bias_initializer,
                          regularizer=self.bias_regularizer,
                          constraint=self.bias_constraint)
        
        """
            Matrices for c (candidate)
        """
        self.V_c = self.add_weight(shape=(self.input_dim,self.units),name='V_c',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        self.W_c = self.add_weight(shape=(self.input_dim_e, self.units),name='W_c',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        self.U_c = self.add_weight(shape=(self.units, self.units),name='U_c',
                                    initializer=self.recurrent_initializer,
                                    regularizer=self.recurrent_regularizer,
                                    constraint=self.recurrent_constraint)
        self.b_c = self.add_weight(shape=(self.units,), name='b_c',
                          initializer=self.bias_initializer,
                          regularizer=self.bias_regularizer,
                          constraint=self.bias_constraint)
        """
            Matrices for mmaxout 
        """
        self.U_p = self.add_weight(shape=(self.units, self.units),name='U_p',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        
        """
            Matrices for making the final prediction vector
        """
        self.M_p = self.add_weight(shape=(self.gmax,self.output_dim),name='M_p',
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        self.b_p = self.add_weight(shape=(self.output_dim,), name='b_p',
                          initializer=self.bias_initializer,
                          regularizer=self.bias_regularizer,
                          constraint=self.bias_constraint)
        
          # For creating the initial state:
        self.W_sc = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_sc',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_sc = self.add_weight(shape=(self.units,), name='b_sc',
                          initializer=self.bias_initializer,
                          regularizer=self.bias_regularizer,
                          constraint=self.bias_constraint)
        
        self.W_sy = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_sy',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_sy = self.add_weight(shape=(self.units,), name='b_sy',
                          initializer=self.bias_initializer,
                          regularizer=self.bias_regularizer,
                          constraint=self.bias_constraint)
        
        self.trainable_weights = [self.C_cx, self.W_cx, self.b_cx,
                                  self.V_i,  self.W_i, self.U_i, self.b_i,
                                  self.V_f,  self.W_f, self.U_f, self.b_f,
                                  self.V_o,  self.W_o, self.U_o, self.b_o,
                                  self.V_c,  self.W_c, self.U_c, self.b_c,
                                  self.M_p,  
                                  self.U_p,  self.b_p,
                                  self.W_sc, self.W_sy, self.b_sc, self.b_sy]
       
        self.input_spec = [
            InputSpec(shape=(self.batch_size_e, self.time_step_e, self.input_dim_e)), InputSpec(shape=(self.batch_size, self.time_step, self.input_dim))]
        self.built = True
        super(attention_LSTM, self).build(input_shape)
     
    def step(self, x, states, training=None):
        h_tm1 = states[0]
        c_tm1 = states[1]
        x_seq = states[2]
        # repeat the hidden state to the length of the sequence
        _htm = K.repeat(h_tm1, self.time_step_e)#(batch,time_step,units)
        # concatenate a(previus output lstm) + hidden state
        concatenate = K.concatenate([_htm,x_seq], axis=-1) #(batch,time_step,h_units+x_seq_units)
        # now multiplty the weight matrix with the repeated hidden state
        
        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        dot_dense = time_distributed_dense(concatenate, self.W_cx, b=self.b_cx,
                                             input_dim=self.input_dim_e + self.units,
                                             timesteps=self.time_step_e,
                                             output_dim=self.atten_units)#(samples,timestep,atten_units)
        # we need to supply the full sequence of inputs to step (as the attention_vector)
        
        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.
        et = K.dot(K.relu(dot_dense), #(batch,time_step,atten_units)
                   K.expand_dims(self.C_cx))
        
        at = K.exp(et)#(batch,time_step,1)
        at_sum = K.cast(K.sum(at, axis=1)+ K.epsilon(), K.floatx())#(batch,1)
        at_sum_repeated = K.repeat(at_sum, self.time_step_e)#(batch,time_step,1)
        at /= at_sum_repeated  # vector of size (batchsize, time_steps, 1)
        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, x_seq, axes=1), axis=1)#(batchsize,input_dim)
        
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(context),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        B_W = self._dropout_mask
        # dropout matrices for recurrent units
        B_U = self._recurrent_dropout_mask
        # ~~~> calculate new hidden state
        
        yhat_i = K.dot(x, self.V_i)#(batchsize,units)
        yhat_f = K.dot(x, self.V_f)#(batchsize,units)
        yhat_c = K.dot(x, self.V_c)#(batchsize,units)
        yhat_o = K.dot(x, self.V_o)#(batchsize,units)
        
        if 0 < self.dropout < 1.:
            x_i = K.dot(context * B_W[0], self.W_i) + self.b_i #(batchsize,units)
            x_f = K.dot(context * B_W[1], self.W_f) + self.b_f#(batchsize,units)
            x_c = K.dot(context * B_W[2], self.W_c) + self.b_c#(batchsize,units)
            x_o = K.dot(context * B_W[3], self.W_o) + self.b_o#(batchsize,units)
        else:
            x_i = K.dot(context , self.W_i) + self.b_i#(batchsize,units)
            x_f = K.dot(context , self.W_f) + self.b_f#(batchsize,units)
            x_c = K.dot(context , self.W_c) + self.b_c#(batchsize,units)
            x_o = K.dot(context , self.W_o) + self.b_o#(batchsize,units)
        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = K.dot(h_tm1 * B_U[0], self.U_i)#(batchsize,units)
            h_tm1_f = K.dot(h_tm1 * B_U[1], self.U_f)#(batchsize,units)
            h_tm1_c = K.dot(h_tm1 * B_U[2], self.U_c)#(batchsize,units)
            h_tm1_o = K.dot(h_tm1 * B_U[3], self.U_o)#(batchsize,units)
        else:
            h_tm1_i = K.dot(h_tm1, self.U_i)#(batchsize,units)
            h_tm1_f = K.dot(h_tm1, self.U_f)#(batchsize,units)
            h_tm1_c = K.dot(h_tm1, self.U_c)#(batchsize,units)
            h_tm1_o = K.dot(h_tm1, self.U_o)#(batchsize,units)
            
        i = self.recurrent_activation(x_i + h_tm1_i+ yhat_i)#(batchsize,units)
        f = self.recurrent_activation(x_f + h_tm1_f + yhat_f)#(batchsize,units)
        o = self.recurrent_activation(x_o + h_tm1_o + yhat_o)#(batchsize,units)
        c_ = self.activation(x_c + h_tm1_c + yhat_c)#(batchsize,units)
        c = f * c_tm1 + i * c_
        
        h = o * self.activation(c)#(batchsize,units)
        
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        #apply maxout layer with dropout
        maxout = self.max_out(inputs= K.dot(h,self.U_p), num_units=self.gmax)
        drop = Dropout(0.3)(maxout)
        #apply softmax
        _y_hat = activations.softmax(K.dot(drop,self.M_p) + self.b_p)
        
        if self.return_probabilities:
            return at, [h, c]
        else:
            return _y_hat, [h, c]
        
    def max_out(self, inputs, num_units, axis=None):
        shape = inputs.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:  # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError('number of features({}) is not '
                             'a multiple of num_units({})'.format(num_channels, num_units))
        shape[axis] = num_units
        shape += [num_channels // num_units]
        print(shape)
        outputs = K.max(tf.reshape(inputs, shape), -1, keepdims=False)
        return outputs

    def prosess_input(self,x):
        return x[:,:self.steps]
    
    def get_constants(self, x):
        constants = []
        # store the whole sequence so we can "attend" to it at each timestep
        constants.append(x)

        return constants
    def get_initial_state(self, inputs):
        print('inputs shape:', inputs.get_shape())
        #y = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        #y = K.sum(y, axis=(1, 2))  # (samples, )
        #y = K.expand_dims(y)  # (samples, 1)
        
        # apply the matrix on the first time step to get the initial c0, s0.
        c0 = self.activation(K.dot(inputs[:, 0], self.W_sc) + self.b_sc) #(samples,units)
        s0 = self.activation(K.dot(inputs[:, 0], self.W_sy)+ self.b_sy)#(samples,units)
        # output_dim)
        #y0 = K.tile(y, [1, self.output_dim])#Repeat 1 in axis=1 and output_dim in axis=2:(samples,output_dim)
        return [s0, c0]
    
    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.units)))
            
            #y0 = K.zeros_like(self.input_spec[0])  # (samples, timesteps, input_dims)
            #y0 = K.sum(y0, axis=(1, 2))  # (samples, )
            #y0 = K.expand_dims(y0)  # (samples, 1)
            #y0 = K.tile(y0, [1, self.output_dim])#(samples,output_dim)
            
            #K.set_value(self.states[2])
        else:
            self.states = [K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], self.units))]
    def call(self, x):
        assert isinstance(x, list)
        enc, dec = x
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
        
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))

      
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_state(dec)
        constants = self.get_constants(enc)
        preprocessed_input = self.prosess_input(dec)
        last_output, outputs, states = K.rnn(self.step, 
                                             preprocessed_input,
                                             initial_states=initial_states, 
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=dec.shape[1])
        
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output
        
    def get_config(self):
        config = {'units': self.units,
                  'steps':self.steps,
                  'output_dim': self.output_dim,
                  'atten_units':self.atten_units,
                  'activation': activations.serialize(self.activation),
                  'gmax':self.gmax,
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'return_probabilities': self.return_probabilities
                 }
        base_config = super(attention_LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
