from dataclasses import dataclass
from typing import List, Optional, Tuple

import tensorflow as tf
from transformers import TFGPT2PreTrainedModel, TFGPT2MainLayer, BatchEncoding
from transformers import AutoTokenizer, TFGPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_tf_outputs import TFCausalLMOutputWithCrossAttentions
from transformers.modeling_tf_utils import input_processing, TFModelInputType, TFCausalLanguageModelingLoss 
from typing import Union
import inspect 
import numpy as np

class TFBalladModel(TFGPT2PreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super(TFBalladModel, self).__init__(config, *inputs, **kwargs)
        self.transformer = TFGPT2MainLayer(config, name="transformer")

    def get_output_embeddings(self):
        return self.get_input_embeddings()

    def set_output_embeddings(self, value):
        self.set_input_embeddings(value)

    def prepare_inputs_for_generation(self, inputs, past=None, use_cache=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            inputs = tf.expand_dims(inputs[:, -1], -1)
            if token_type_ids is not None:
                token_type_ids = tf.expand_dims(token_type_ids[:, -1], -1)

        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            if past:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)

        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past": past,
            "use_cache": use_cache,
            "token_type_ids": token_type_ids,
        }

    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        past: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_hidden_states: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
        training: Optional[bool] = False,
    ) -> Union[TFCausalLMOutputWithCrossAttentions, Tuple[tf.Tensor]]:

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        hidden_states = transformer_outputs[0]
        logits = self.transformer.wte(hidden_states, mode="linear")

        loss = None
        if labels is not None:
            # shift labels to the left and cut last logit token
            shifted_logits = logits[:, :-1]
            labels = labels[:, 1:]
            loss = self.hf_compute_loss(labels, shifted_logits)

        # OUR CODE HERE -> UPDATE THE LOSS CALCULATION; need to find if the words at the end of each line rhyme...
        # right now the "loss" is only computed given labels and logits

        # END CODE

        #if not return_dict:
        #    output = (logits,) + transformer_outputs[1:]
        #    return ((loss,) + output) if loss is not None else output

        return TFCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def serving_output(self, output):
        pkv = tf.convert_to_tensor(output.past_key_values) if self.config.use_cache else None
        hs = tf.convert_to_tensor(output.hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.attentions) if self.config.output_attentions else None
        cross_attns = (
            tf.convert_to_tensor(output.cross_attentions)
            if self.config.output_attentions
            and self.config.add_cross_attention
            and output.cross_attentions is not None
            else None
        )

        return TFCausalLMOutputWithCrossAttentions(
            logits=output.logits, past_key_values=pkv, hidden_states=hs, attentions=attns, cross_attentions=cross_attns
        )

def load_model_and_tokenizer(path_to_weights="gpt2ballads_params/gpt2-ep1"):
  MAX_TOKENS = 64
  BOS_TOKEN = "<|beginoftext|>"
  EOS_TOKEN = "<|endoftext|>"
  PAD_TOKEN = "<|pad|>"

  # this will download and initialize the pre trained tokenizer
  tokenizer = GPT2Tokenizer.from_pretrained(
      "gpt2",
      bos_token=BOS_TOKEN,
      eos_token=EOS_TOKEN,
      pad_token=PAD_TOKEN,
      max_length=MAX_TOKENS,
      is_split_into_words=True,
  )
  print(len(tokenizer))

  EPOCHS = 10
  INITIAL_LEARNING_RATE = 0.001
  DECAY_STEPS = 300
  DECAY_RATE = 0.7

  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      INITIAL_LEARNING_RATE,
      decay_steps=DECAY_STEPS,
      decay_rate=DECAY_RATE,
      staircase=True)

  model = TFBalladModel.from_pretrained(
          "gpt2",
          use_cache=False,
          pad_token_id=tokenizer.pad_token_id,
          eos_token_id=tokenizer.eos_token_id,
          bos_token_id=tokenizer.bos_token_id
      )
  model.resize_token_embeddings(len(tokenizer))
  #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  model.compile()
  model.layers[0].vocab_size = len(tokenizer) # something is wrong with TFGPT2 initialization so this is needed
  #model.summary()
  # load the weights now
  model.load_weights(path_to_weights) # this is the path to the unpacked weights, should not be changed (unless you changed the output folder in the unpacking line)
  return model, tokenizer

def generate_ballad_lines(model, tokenizer, keywords, n=4):
  lines = []
  while len(lines) != n: # hard coded for the generator to produce 4 lines of text
    prompt = "<|beginoftext|> Keywords: " + " ".join(keywords) + "\nBallad: "
    input_ids = tokenizer.encode(prompt, return_tensors='tf')
    sample_output = model.generate(input_ids, do_sample=True, max_length=64, top_k=50, top_p=0.95, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True).strip()
    lines = generated_text.split("\n")[1:]
    lines[0] = lines[0][len("Ballad: "):].strip()
    lines = [line for line in lines if len(line) > 0]
  return lines