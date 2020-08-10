{
  "seed" : 0,
  "data" : {
    "dataset" : "iamdb",
    "data_path" : "/datasets01/iamdb/060820/",
    "img_height" : 64
  },
  "criterion_type" : "transducer",
  "criterion" : {
    "blank" : false,
    "allow_repeats" : true,
    "transitions" : "/checkpoint/awni/data/iamdb/transitions_grapheme_p0_10.txt"
  },
  "model_type" : "tds2d",
  "model" : {
    "depth" : 4,
    "tds_groups" : [
      { "channels" : 4, "num_blocks" : 3, "stride" : [2, 2] },
      { "channels" : 16, "num_blocks" : 3, "stride" : [2, 2] },
      { "channels" : 32, "num_blocks" : 3, "stride" : [2, 1] },
      { "channels" : 64, "num_blocks" : 3, "stride" : [2, 1] }
    ],
    "kernel_size" : [5, 7],
    "dropout" : 0.1
  },
  "optim" : {
    "batch_size" : 32,
    "epochs" : 400,
    "learning_rate" : 1e-1,
    "crit_learning_rate" : 1e-1,
    "step_size" : 100,
    "max_grad_norm" : 5
  }
}
