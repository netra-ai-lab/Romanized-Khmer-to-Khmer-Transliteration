from __future__ import annotations

from tensorflow.keras.layers import (
    AdditiveAttention,
    Bidirectional,
    Concatenate,
    Dense,
    Embedding,
    GRU,
    Input,
    TimeDistributed,
)
from tensorflow.keras.models import Model


def build_inference_models(model) -> tuple:
    """Deconstruct a trained BiGRU-Attention model into separate encoder and decoder inference models.

    Returns:
        (encoder_model, decoder_model, gru_units)
    """
    encoder_embedding = None
    decoder_embedding = None
    encoder_gru = None
    decoder_gru = None
    encoder_proj_dense = None
    attention_layer = None
    decoder_dense = None

    for layer in model.layers:
        if isinstance(layer, Embedding):
            if encoder_embedding is None:
                encoder_embedding = layer
            else:
                decoder_embedding = layer
        elif isinstance(layer, Bidirectional):
            encoder_gru = layer
        elif isinstance(layer, GRU):
            decoder_gru = layer
        elif isinstance(layer, TimeDistributed):
            encoder_proj_dense = layer
        elif isinstance(layer, AdditiveAttention):
            attention_layer = layer
        elif isinstance(layer, Dense) and layer.activation.__name__ == "softmax":
            decoder_dense = layer

    # Encoder inference model
    encoder_inputs = Input(shape=(None,))
    enc_emb = encoder_embedding(encoder_inputs)
    encoder_outputs = encoder_gru(enc_emb)
    encoder_proj = encoder_proj_dense(encoder_outputs)
    encoder_model = Model(encoder_inputs, [encoder_outputs, encoder_proj])

    # Decoder inference model
    gru_units = decoder_gru.units
    decoder_inputs = Input(shape=(None,))
    decoder_state_input = Input(shape=(gru_units,))
    encoder_proj_input = Input(shape=(None, gru_units))

    dec_emb = decoder_embedding(decoder_inputs)
    decoder_outputs_inf, decoder_state_inf = decoder_gru(dec_emb, initial_state=decoder_state_input)
    context_inf = attention_layer([decoder_outputs_inf, encoder_proj_input])
    decoder_combined_inf = Concatenate(axis=-1)([decoder_outputs_inf, context_inf])
    decoder_outputs_final_inf = decoder_dense(decoder_combined_inf)

    decoder_model = Model(
        [decoder_inputs, decoder_state_input, encoder_proj_input],
        [decoder_outputs_final_inf, decoder_state_inf],
    )

    return encoder_model, decoder_model, gru_units
