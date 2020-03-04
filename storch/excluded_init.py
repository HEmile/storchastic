
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
_unwrap_only = {
    "_has_compatible_shallow_copy_type",
    "empty_like",
    "full_like",
    "ones_like",
    "is_distributed",
    "is_floating_point",
    "is_same_size",
    "cudnn_is_acceptable",
    "numel",
    "result_type",
    "zeros_like"
}
_exception_init = {
    "allclose",
    "equal",
    "is_complex",
    "is_nonzero",
    "is_signed",
    "q_per_channel_axis",
    "q_scale",
    "q_zero_point"
}
_to_test = {
    "allclose",
    "cudnn_is_acceptable",
    "fbgemm_linear_quantize_weight",
    "numel"
}
# print(_excluded_init)