import onnx
import numpy as np
from onnx import helper, TensorProto

def create_rnn_family_onnx_model(operator_type="RNN", sequence_length=3, batch_size=2, input_size=4, hidden_size=5,
                                direction="forward", activation=None, activations=None,
                                include_sequence_lens=False, include_initial_h=False, include_initial_c=False):
    """
    Create a single-node ONNX model for RNN family operators (RNN, GRU, LSTM)
    """
    num_directions = 2 if direction == "bidirectional" else 1

    # Define gate counts and default activations for each operator
    operator_configs = {
        "RNN": {"gates": 1, "default_activations": ["Tanh"]},
        "GRU": {"gates": 3, "default_activations": ["Sigmoid", "Tanh"]},
        "LSTM": {"gates": 4, "default_activations": ["Sigmoid", "Tanh", "Tanh"]}
    }

    if operator_type not in operator_configs:
        raise ValueError(f"Unsupported operator type: {operator_type}")

    config = operator_configs[operator_type]
    gates = config["gates"]

    # Set activations according to ONNX specification
    if activations is None:
        if operator_type == "RNN":
            # For RNN, use the provided activation (could be Tanh or Relu), default to Tanh if None
            if activation is None:
                activation = "Tanh"
            activations = [activation] * num_directions
        elif operator_type == "GRU":
            # For GRU, need 2 activations per direction: [Sigmoid, Tanh]
            base_activations = ["Sigmoid", "Tanh"]
            activations = base_activations * num_directions
        elif operator_type == "LSTM":
            # For LSTM, need 3 activations per direction: [Sigmoid, Tanh, Tanh]
            base_activations = ["Sigmoid", "Tanh", "Tanh"]
            activations = base_activations * num_directions

    # Create input value infos
    X_shape = [sequence_length, batch_size, input_size]
    W_shape = [num_directions, gates * hidden_size, input_size]
    R_shape = [num_directions, gates * hidden_size, hidden_size]
    B_shape = [num_directions, 2 * gates * hidden_size]

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, X_shape)
    W = helper.make_tensor_value_info('W', TensorProto.FLOAT, W_shape)
    R = helper.make_tensor_value_info('R', TensorProto.FLOAT, R_shape)
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, B_shape)

    inputs = [X, W, R, B]
    node_inputs = ['X', 'W', 'R', 'B']

    # Handle sequence_lens (5th input)
    if include_sequence_lens:
        sequence_lens_shape = [batch_size]
        sequence_lens = helper.make_tensor_value_info('sequence_lens', TensorProto.INT32, sequence_lens_shape)
        inputs.append(sequence_lens)
        node_inputs.append('sequence_lens')
    else:
        node_inputs.append('')

    # Handle initial_h (6th input)
    if include_initial_h:
        initial_h_shape = [num_directions, batch_size, hidden_size]
        initial_h = helper.make_tensor_value_info('initial_h', TensorProto.FLOAT, initial_h_shape)
        inputs.append(initial_h)
        node_inputs.append('initial_h')
    else:
        node_inputs.append('')

    # Handle initial_c (7th input, LSTM only)
    if operator_type == "LSTM":
        if include_initial_c:
            initial_c_shape = [num_directions, batch_size, hidden_size]
            initial_c = helper.make_tensor_value_info('initial_c', TensorProto.FLOAT, initial_c_shape)
            inputs.append(initial_c)
            node_inputs.append('initial_c')
        else:
            node_inputs.append('')

    # Create output value infos
    Y_shape = [sequence_length, num_directions, batch_size, hidden_size]
    Y_h_shape = [num_directions, batch_size, hidden_size]

    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, Y_shape)
    Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, Y_h_shape)

    outputs = [Y, Y_h]
    node_outputs = ['Y', 'Y_h']

    # Add Y_c for LSTM
    if operator_type == "LSTM":
        Y_c_shape = [num_directions, batch_size, hidden_size]
        Y_c = helper.make_tensor_value_info('Y_c', TensorProto.FLOAT, Y_c_shape)
        outputs.append(Y_c)
        node_outputs.append('Y_c')

    # Create node attributes
    node_attrs = {
        'direction': direction,
        'hidden_size': hidden_size,
        'activations': activations
    }

    # Create the operator node
    node = helper.make_node(
        operator_type,
        inputs=node_inputs,
        outputs=node_outputs,
        **node_attrs
    )

    # Create graph
    graph = helper.make_graph(
        [node],
        f'{operator_type.lower()}_model',
        inputs,
        outputs
    )

    # Create model with compatible IR version and opset
    # Use IR version 7 and opset version 11 for better compatibility
    model = helper.make_model(
        graph,
        producer_name='rnn_family_test',
        ir_version=7,  # Compatible IR version
        opset_imports=[helper.make_opsetid("", 11)]  # Compatible opset version
    )

    return model

# Keep original function for backward compatibility
def create_rnn_onnx_model(*args, **kwargs):
    return create_rnn_family_onnx_model(operator_type="RNN", *args, **kwargs)

def save_rnn_family_model(operator_type="RNN", **kwargs):
    """Save RNN family model with proper filename generation"""
    model = create_rnn_family_onnx_model(operator_type=operator_type, **kwargs)

    # Generate filename
    direction = kwargs.get('direction', 'forward')
    activation = kwargs.get('activation', 'Tanh')
    include_sequence_lens = kwargs.get('include_sequence_lens', False)
    include_initial_h = kwargs.get('include_initial_h', False)
    include_initial_c = kwargs.get('include_initial_c', False)

    filename = f'{operator_type.lower()}_model_{direction}_{activation.lower()}'
    if include_sequence_lens:
        filename += '_with_seq_lens'
    if include_initial_h:
        filename += '_with_initial_h'
    if include_initial_c and operator_type == "LSTM":
        filename += '_with_initial_c'
    filename += '.onnx'

    onnx.save(model, filename)
    print(f"{operator_type} model saved as {filename}")
    return model

def save_rnn_model(**kwargs):
    """Save RNN model with proper filename generation (backward compatibility)"""
    return save_rnn_family_model(operator_type="RNN", **kwargs)

def create_test_models():
    """Create comprehensive test models demonstrating ONNX RNN specification compliance"""
    print("=" * 60)
    print("Creating ONNX RNN Family Test Models")
    print("ONNX RNN Specification:")
    print("- sequence_lens: Optional, default = seq_length for all sequences")
    print("- initial_h: Optional, default = zeros")
    print("- initial_c: Optional for LSTM, default = zeros")
    print("=" * 60)

    # Test all RNN family operators
    operators = ["RNN", "GRU", "LSTM"]

    for operator_type in operators:
        print(f"\n=== Creating {operator_type} Models ===")

        # Test configurations for each operator
        configs = [
            # Basic models with different directions and activations
            {"direction": "forward", "activation": "Tanh", "seq_lens": False, "initial_h": False, "initial_c": False,
             "desc": f"Forward Tanh with ONNX defaults"},
            {"direction": "forward", "activation": "Tanh", "seq_lens": True, "initial_h": True, "initial_c": False,
             "desc": f"Forward with optional inputs"},
        ]

        # Add LSTM-specific test with initial_c
        if operator_type == "LSTM":
            configs.append({
                "direction": "forward", "activation": "Tanh", "seq_lens": True, "initial_h": True, "initial_c": True,
                "desc": f"LSTM with all optional inputs including initial_c"
            })

        # Add RNN-specific activation test
        if operator_type == "RNN":
            configs.append({
                "direction": "forward", "activation": "Relu", "seq_lens": False, "initial_h": False, "initial_c": False,
                "desc": f"RNN Forward ReLU with ONNX defaults"
            })

        models = []
        for i, config in enumerate(configs, 1):
            print(f"\nModel {i}: {config['desc']}")
            try:
                model = save_rnn_family_model(
                    operator_type=operator_type,
                    include_sequence_lens=config["seq_lens"],
                    include_initial_h=config["initial_h"],
                    include_initial_c=config["initial_c"] and operator_type == "LSTM",
                    direction=config["direction"],
                    activation=config["activation"]
                )
                models.append(model)
                print("✓ Model created successfully")
            except Exception as e:
                print(f"✗ Model creation failed: {e}")

        print(f"\n✓ Created {len(models)} {operator_type} models")

    return models

if __name__ == "__main__":
    import sys
    
    # Check if we have command line arguments (excluding script name)
    if len(sys.argv) > 1:
        # Command line mode
        import argparse
        
        parser = argparse.ArgumentParser(description='Create ONNX RNN models')
        parser.add_argument('--operator_type', type=str, default='RNN', choices=['RNN', 'GRU', 'LSTM'])
        parser.add_argument('--sequence_length', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--input_size', type=int, default=4)
        parser.add_argument('--hidden_size', type=int, default=5)
        parser.add_argument('--direction', type=str, default='forward', choices=['forward', 'bidirectional'])
        parser.add_argument('--activation', type=str, default='Tanh', choices=['Tanh', 'ReLU', 'Sigmoid', 'Relu'])
        parser.add_argument('--include_sequence_lens', type=int, default=0)
        parser.add_argument('--include_initial_h', type=int, default=0)
        parser.add_argument('--include_initial_c', type=int, default=0)
        parser.add_argument('--output_filename', type=str, required=True)
        
        args = parser.parse_args()
        
        # Normalize activation name
        activation = args.activation
        if activation == 'Relu':
            activation = 'ReLU'
        
        # Create single model with specified parameters
        # For GRU and LSTM, don't pass activation parameter to use default activations
        if args.operator_type == "RNN":
            model = create_rnn_family_onnx_model(
                operator_type=args.operator_type,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                direction=args.direction,
                activation=activation,
                include_sequence_lens=args.include_sequence_lens == 1,
                include_initial_h=args.include_initial_h == 1,
                include_initial_c=args.include_initial_c == 1
            )
        else:
            # For GRU and LSTM, use None as activation to get default activations
            model = create_rnn_family_onnx_model(
                operator_type=args.operator_type,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                direction=args.direction,
                activation=None,  # Use default activations for GRU/LSTM
                include_sequence_lens=args.include_sequence_lens == 1,
                include_initial_h=args.include_initial_h == 1,
                include_initial_c=args.include_initial_c == 1
            )
        
        # Save the model
        onnx.save(model, args.output_filename)
        print(f"Model saved as {args.output_filename}")
    else:
        # Test mode - create comprehensive test models
        models = create_test_models()
        
        print("\n" + "=" * 60)
        print("Summary:")
        print("- Models without optional inputs use ONNX defaults:")
        print("  * sequence_lens = seq_length for all sequences")
        print("  * initial_h = zeros")
        print("- Models with optional inputs allow custom values")
        print("- All models follow ONNX RNN specification")
