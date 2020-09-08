import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
#%%
inputs = tf.random.normal([32, 10, 8])
lstm = tf.keras.layers.LSTM(4, activation='tanh')
output = lstm(inputs)
print(output.shape)
#%%
lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
print(whole_seq_output.shape)

print(final_memory_state.shape)

print(final_carry_state.shape)

