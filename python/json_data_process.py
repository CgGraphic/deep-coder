import numpy as np
import settings as parameters

def convert_integer(integer):
    return integer -parameters.integer_min # integer_min -> 0
def type_vector(value):
    if isinstance(value, list):
        return [0, 1]
    elif isinstance(value, int):
        return [1, 0]
    else:
        return [0, 0]
def convert_value(value):
    # type vector
    t = type_vector(value)
    if isinstance(value, int):
        value = [value]
    elif isinstance(value, list):
        value = value
    else:
        value = []
    value = [convert_integer(x) for x in value]
    # Fill NULL (integer_range)
    if len(value) < parameters.list_length:
        add = [parameters.integer_range] * (parameters.list_length - len(value))
        value.extend(add)
    t.extend(value)
    return np.array(t, dtype=np.float32)

def convert_example(example):
    # Fill NULL input
    input = example['input']
    if len(input) < parameters.input_num:
        add = [""] * (parameters.input_num - len(input))
        input.extend(add)
    output = example['output']
    x = [convert_value(y) for y in input]
    x.extend([convert_value(output)])
    return np.array(x)

def convert_each_data(data):
    examples = data['examples']
    # Convert
    examples2 = np.array([convert_example(x) for x in examples])
    attrs = np.array(data['attribute'], dtype=np.int32)
    return examples2, attrs

def preprocess_json(data):
    return [convert_each_data(x) for x in data]
