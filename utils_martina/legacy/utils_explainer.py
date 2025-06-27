import json

################################################################################################################

def create_explainer_json(epochs=1000, lr=0.001, discriminator='SimpleDiscriminator'):

    # Set explainer json file

    file_json = {  
        "experiment" : {
            "scope": "examples_configs",
            "parameters" : {
                "lock_release_tout":120,
                "propagate": [
                    {"in_sections": ["explainers"], "params": {"fold_id": 0}},
                    {"in_sections": ["do-pairs/oracle"], "params": {"fold_id": -1,"retrain":True}},
                    {"in_sections": ["do-pairs/dataset"], "params": { "compose_man": "config/snippets/datasets/EEG.json" }}
                ]
            }
        },
        "compose_mes": "config/snippets/default_metrics.json",
        "compose_strs": "config/snippets/default_store_paths.json",
        "do-pairs": [ {"compose_bbbp_gcn" : "config/snippets/do-pairs/EEG-CORR.json"} ],
        "explainers": [
            {
                "class": "src.explainer.generative.rsgg.RSGG",
                "parameters": {
                    "fold_id": 0,
                    "retrain": True,
                    "epochs": epochs,
                    "sampler": {
                        "class": "src.utils.samplers.partial_order_samplers.PositiveAndNegativeEdgeSampler",
                        "parameters": {"sampling_iterations": 500}
                    },
                    "models": [
                    {
                        "class": "src.explainer.generative.gans.graph.model.GAN",
                        "parameters": {
                            "model_label": 0,
                            "fold_id": 0,
                            "retrain": True,
                            "epochs": epochs,
                            "batch_size": 1,
                            "loss_fn": {"class": "torch.nn.BCELoss", "parameters": {"reduction": "mean"}},
                            "generator": {
                                "class": "src.explainer.generative.gans.graph.res_gen.ResGenerator",
                                "parameters": {}
                            },
                            "discriminator": {
                                "class": f"src.explainer.generative.gans.graph.discriminators.{discriminator}",
                                "parameters": {}
                            },
                            "gen_optimizer": {
                                "class": "torch.optim.SGD",
                                "parameters": {
                                    "lr": lr,
                                    "momentum": 0,
                                    "dampening": 0,
                                    "weight_decay": 0,
                                    "nesterov": False,
                                    "maximize": False,
                                    "differentiable": False
                                }
                            },
                            "disc_optimizer": {
                                "class": "torch.optim.SGD",
                                "parameters": {
                                    "lr": lr,
                                    "momentum": 0,
                                    "dampening": 0,
                                    "weight_decay": 0,
                                    "nesterov": False,
                                    "maximize": False,
                                    "differentiable": False
                                }
                            }
                        }
                    },
                    {
                        "class": "src.explainer.generative.gans.graph.model.GAN",
                        "parameters": {
                            "model_label": 1,
                            "fold_id": 0,
                            "retrain": True,
                            "epochs": epochs,
                            "batch_size": 1,
                            "loss_fn": {"class": "torch.nn.BCELoss", "parameters": {"reduction": "mean"}},
                            "generator": {
                                "class": "src.explainer.generative.gans.graph.res_gen.ResGenerator",
                                "parameters": {}
                            },
                            "discriminator": {
                                "class": f"src.explainer.generative.gans.graph.discriminators.{discriminator}",
                                "parameters": {}
                            },
                            "gen_optimizer": {
                                "class": "torch.optim.SGD",
                                "parameters": {
                                    "lr": lr,
                                    "momentum": 0,
                                    "dampening": 0,
                                    "weight_decay": 0,
                                    "nesterov": False,
                                    "maximize": False,
                                    "differentiable": False
                                }
                            },
                            "disc_optimizer": {
                                "class": "torch.optim.SGD",
                                "parameters": {
                                    "lr": lr,
                                    "momentum": 0,
                                    "dampening": 0,
                                    "weight_decay": 0,
                                    "nesterov": False,
                                    "maximize": False,
                                    "differentiable": False
                                }
                            }
                        }
                    }
                    ]
                }
            }
        ]
    }

    with open("config\EEG-CORR.jsonc", "w") as f:
        json.dump(file_json, f, indent=4)