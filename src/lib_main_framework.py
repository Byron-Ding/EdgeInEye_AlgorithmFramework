import inspect
from functools import reduce
from inspect import Parameter
from typing import Any, Union
from typing import Callable

import numpy
from torch import Tensor



class MatPositionMultFramework:
    def __new__(
        cls,
        *fns: Callable[[], Any],
        **named_fns: Callable[[], Any],
    ) -> Union[numpy.ndarray, int]:
        # ------------------ Extract and validate functions ------------------
        # 解包，非空验证
        all_fns = [*fns, *named_fns.values()]   # 位置 + 命名，都当零参函数

        if not all_fns:
            raise ValueError("At least one function must be provided.")


        # ------------------ Execute functions and validate shapes ------------------
        results = [fn() for fn in all_fns]      # 仍是零参调用

        numpy.broadcast_shapes(*[numpy.shape(r) for r in results])
        return reduce(numpy.multiply, results)


    @staticmethod
    def _verify_zero_args(fn) -> bool:
        """True = 该函数仍需调用方提供参数。对 partial/闭包同样有效。"""

        # ------------------ Check if the function is a built-in or has no functions ------------------
        try:
            fn_sig: inspect.Signature = inspect.signature(fn)
        except (ValueError, TypeError):
            # Builtin OR no signature, assume it requires arguments
            return False
        except Exception as e:
            raise RuntimeError(f"Error inspecting function {fn}: {e}")


        # ------------------ Check if all parameters have default values ------------------
        func_params = fn_sig.parameters.values()
        # All rest parameters must have default values
        return all(
            p.default is not Parameter.empty  # Default value exists
            or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)  # Accepts *args or **kwargs (accecpt missing args)
            for p in func_params
        )


class MultiFuncResultFramework(MatPositionMultFramework):
    def __new__(cls,
        *fns: Callable[..., Union[float, Tensor, numpy.ndarray]],
        threshold_function: Callable[..., Union[float, Tensor, numpy.ndarray]],
        compare_function: Callable[..., Union[float, Tensor, numpy.ndarray]],
        weight_function: Callable[..., Union[float, Tensor, numpy.ndarray]],
        **named_fns: Any,
    ) -> Union[float, Tensor, numpy.ndarray]:
        """
        A framework function that applies a threshold function, a comparison function, and a weight function
        to the provided arguments.

        Output = F_{\text{Threshold}}(
                F_{\text{Localization}_{convolutionHere}}(
                    F_{\text{Compare}}
                        \\cdot F_{\text{Weight}_{Distribution}}
                        \\cdot F_{\text{Direction}}
                )
            )


        Parameters:
        - threshold_function: A callable that computes a threshold value.
        - compare_function: A callable that compares values based on the threshold.
        - weight_function: A callable that computes weights for the comparison.
        - *args: Positional arguments to be passed to the functions.
        - **kwargs: Keyword arguments to be passed to the functions.

        Returns:
        - The result of applying the comparison and weighting based on the threshold.
        """

        # Compute the inner result
        # Call the father query function
        weighted_result: Union[float, Tensor, numpy.ndarray] = super().__new__(
            cls,
            threshold_function,
            compare_function,
            weight_function,
            *fns,
            **named_fns
        )  # This will compute the product of all fns

        return weighted_result








