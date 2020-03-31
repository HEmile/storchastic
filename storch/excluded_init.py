_excluded_init = {
    "_debug_has_internal_overlap",
    "_nnpack_available",
    "_use_cudnn_ctc_loss",
    "can_cast",
    "empty",
    "empty_strided",
    "from_file",
    "from_numpy",
    "full",
    "get_default_dtype",
    "get_num_interop_threads",
    "get_num_threads",
    "hamming_window",
    "hann_window",
    "logspace",
    "ones",
    "promote_types",
    "range",
    "tril_indices",
    "triu_indices",
    "set_flush_denormal",
    "set_num_interop_threads",
    "set_num_threads",
    "zeros",
    "compiled_with_cxx11_abi",
}
_unwrap_only_init = {
    "_has_compatible_shallow_copy_type",
    "empty_like",
    "full_like",
    "ones_like",
    "is_distributed",
    "is_floating_point",
    "is_signed",
    "is_same_size",
    "cudnn_is_acceptable",
    "numel",
    "result_type",
    "zeros_like",
}
_exception_init = {
    "allclose",
    "equal",
    "is_complex",
    "is_nonzero",
    "q_per_channel_axis",
    "q_scale",
    "q_zero_point",
}
_to_test_init = {
    "allclose",
    "cudnn_is_acceptable",
    "fbgemm_linear_quantize_weight",
    "numel",
}

_excluded_function = {"get_softmax_dim", "assert_int-or-pair"}

_excluded_tensor = {"__getitem__", "__setitem__"}

_exception_tensor = {
    "allclose",
    "bernoulli",
    "bernoulli_",
    "equal",
    "is_complex",
    "is_nonzero",
    "item",
    "normal_" "q_per_channel_axis",
    "q_scale",
    "q_zero_point",
    "random_",
    "stride",
    "to_list",
    "tolist",
    "uniform_",
}

_unwrap_only_tensor = {
    "_make_subclass" "_dimI",
    "_dimV",
    "_is_view",
    "_nnz",
    "dense_dim",
    "get_device",
    "has_names",
    "is_coalesced",
    "is_continguous",
    "is_distributed",
    "is_floating_point",
    "is_pinned",
    "is_same_size",
    "is_set_to",
    "is_signed",
    "nelement",
    "numel",
    "qscheme",
    "size",
    "sparse_dim",
    "storage",
    "storage_offset",
    "new_tensor",
    "new_zeros",
    "new_ones",
}

_to_test_tensor = {"_make_subclass"}
# print(_excluded_init)
