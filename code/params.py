PARAMETERS = {
    # Language by default
    "default_language":"es",

    # Languages used by the system
    "languages": ["es", "en"],

    # Transformers models matching the language
    "transformers_by_language":{
        "es":"",
        "en":""
    },

    # Training parameters --------------------------------------------------------
    "training_params_by_language":{
        "es":{
            # * Learning rate hyperparameter used by default in training process
            "lr": 5e-5,

            # * This value is multiplied to the learning rate powered to the layer id in the layers of the tranformer model
            # * > lr_{layer_i} = lr * (lr_factor ^ i)
            "lr_factor":9/10,

            # * Weight decay used in hte encoder optimizer
            "encoder_decay": 2e-5,

            # * Number of epochs in the training process
            "epochs": 15,

            # * Algorithm used to train the encoder model ["adam", "rms"]
            "encoder_optimizer": "adam",

            # * Weights for unbalance data
            "training_weights":[0.5,0.5],

            # * Transformer embedding_size
            "transformer_embedding_size": 768,

            # * maximung lenght of the input sequence
            "max_lenght": 120,

            # * target frequency change
            "target_frequency": 1,
        },
        "en":{}
    }
}