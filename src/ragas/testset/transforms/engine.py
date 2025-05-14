# from __future__ import annotations

# import asyncio
# import logging
# import typing as t

# from tqdm.auto import tqdm

# from ragas.executor import as_completed, is_event_loop_running
# from ragas.run_config import RunConfig
# from ragas.testset.graph import KnowledgeGraph
# from ragas.testset.transforms.base import BaseGraphTransformation

# if t.TYPE_CHECKING:
#     from langchain_core.callbacks import Callbacks

# logger = logging.getLogger(__name__)

# Transforms = t.Union[
#     t.List[BaseGraphTransformation],
#     "Parallel",
#     BaseGraphTransformation,
# ]


# class Parallel:
#     """
#     Collection of transformations to be applied in parallel.

#     Examples
#     --------
#     >>> Parallel(HeadlinesExtractor(), SummaryExtractor())
#     """

#     def __init__(self, *transformations: BaseGraphTransformation):
#         self.transformations = list(transformations)

#     def generate_execution_plan(self, kg: KnowledgeGraph) -> t.List[t.Coroutine]:
#         coroutines = []
#         for transformation in self.transformations:
#             coroutines.extend(transformation.generate_execution_plan(kg))
#         return coroutines


# async def run_coroutines(
#     coroutines: t.List[t.Coroutine], desc: str, max_workers: int, callbacks: t.Optional[Callbacks] = None
# ):
#     """
#     Run a list of coroutines in parallel with optional callbacks.
#     """
#     for future in tqdm(
#         await as_completed(coroutines, max_workers=max_workers),
#         desc=desc,
#         total=len(coroutines),
#         leave=False,
#     ):
#         try:
#             result = await future
#             # Trigger callbacks if provided
#             if callbacks:
#                 for callback in callbacks:
#                     # Safely call on_success if it exists
#                     if hasattr(callback, 'on_success'):
#                         try:
#                             callback.on_success(result)
#                         except Exception as e:
#                             logger.error(f"Error in callback on_success: {e}")
#         except Exception as e:
#             if callbacks:
#                 for callback in callbacks:
#                     # Safely call on_error if it exists
#                     if hasattr(callback, 'on_error'):
#                         try:
#                             callback.on_error(e)
#                         except Exception as callback_error:
#                             logger.error(f"Error in callback on_error: {callback_error}")
#             logger.error(f"Unable to apply transformation: {e}")


# def get_desc(transform: BaseGraphTransformation | Parallel):
#     if isinstance(transform, Parallel):
#         transform_names = [t.__class__.__name__ for t in transform.transformations]
#         return f"Applying [{', '.join(transform_names)}]"
#     else:
#         return f"Applying {transform.__class__.__name__}"


# def get_desc(transform: BaseGraphTransformation | Parallel):
#     if isinstance(transform, Parallel):
#         transform_names = [t.__class__.__name__ for t in transform.transformations]
#         return f"Applying [{', '.join(transform_names)}]"
#     else:
#         return f"Applying {transform.__class__.__name__}"


# def apply_nest_asyncio():
#     NEST_ASYNCIO_APPLIED: bool = False
#     if is_event_loop_running():
#         # an event loop is running so call nested_asyncio to fix this
#         try:
#             import nest_asyncio
#         except ImportError:
#             raise ImportError(
#                 "It seems like your running this in a jupyter-like environment. Please install nest_asyncio with `pip install nest_asyncio` to make it work."
#             )

#         if not NEST_ASYNCIO_APPLIED:
#             nest_asyncio.apply()
#             NEST_ASYNCIO_APPLIED = True

# from ragas.cost import CostCallbackHandler, get_token_usage_for_openai

# from ragas.callbacks import new_group

# def apply_transforms(
#     kg: KnowledgeGraph,
#     transforms: Transforms,
#     run_config: RunConfig = RunConfig(),
#     callbacks: t.Optional[Callbacks] = None,
#     cost_per_input_token: float = 0.01,
#     cost_per_output_token: float = 0.02,
# ):
#     """
#     Apply a list of transformations to a knowledge graph in place and estimate the cost.
#     """
#     # Initialize callbacks if not provided
#     callbacks = callbacks or []
    
#     # Add a cost callback handler for tracking token usage
#     cost_cb = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
#     callbacks.append(cost_cb)
    
#     # Apply nest_asyncio to fix the event loop issue in Jupyter
#     apply_nest_asyncio()
    
#     # If single transformation, wrap it in a list
#     if isinstance(transforms, BaseGraphTransformation):
#         transforms = [transforms]
    
#     # Apply the transformations
#     if isinstance(transforms, t.List):
#         for transform in transforms:
#             # Check if the transform has the callbacks attribute before trying to access it
#             if hasattr(transform, 'callbacks'):
#                 if transform.callbacks is None and callbacks:
#                     transform.callbacks = callbacks
            
#             asyncio.run(
#                 run_coroutines(
#                     transform.generate_execution_plan(kg),
#                     get_desc(transform),
#                     run_config.max_workers,
#                     callbacks=callbacks  # Pass callbacks here
#                 )
#             )
#     elif isinstance(transforms, Parallel):
#         asyncio.run(
#             run_coroutines(
#                 transforms.generate_execution_plan(kg),
#                 get_desc(transforms),
#                 run_config.max_workers,
#                 callbacks=callbacks  # Pass callbacks here
#             )
#         )
#     else:
#         raise ValueError(
#             f"Invalid transforms type: {type(transforms)}. Expects a list of BaseGraphTransformations or a Parallel instance."
#         )
    
#     # Add a fallback in case no token usage data was collected
#     if not hasattr(cost_cb, 'usage_data') or not cost_cb.usage_data:
#         print("Warning: No token usage data was collected. Returning 0 cost.")
#         return 0.0
        
#     # Calculate and return the total cost
#     total_cost = cost_cb.total_cost(
#         cost_per_input_token=cost_per_input_token,
#         cost_per_output_token=cost_per_output_token,
#     )
#     return total_cost


# def rollback_transforms(kg: KnowledgeGraph, transforms: Transforms):
#     """
#     Rollback a list of transformations from a knowledge graph.

#     Note
#     ----
#     This is not yet implemented. Please open an issue if you need this feature.
#     """
#     # this will allow you to roll back the transformations
#     raise NotImplementedError


from __future__ import annotations

import asyncio
import logging
import typing as t

from tqdm.auto import tqdm

from ragas.executor import as_completed, is_event_loop_running
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.transforms.base import BaseGraphTransformation

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)

Transforms = t.Union[
    t.List[BaseGraphTransformation],
    "Parallel",
    BaseGraphTransformation,
]


class Parallel:
    """
    Collection of transformations to be applied in parallel.

    Examples
    --------
    >>> Parallel(HeadlinesExtractor(), SummaryExtractor())
    """

    def __init__(self, *transformations: BaseGraphTransformation):
        self.transformations = list(transformations)

    def generate_execution_plan(self, kg: KnowledgeGraph) -> t.List[t.Coroutine]:
        coroutines = []
        for transformation in self.transformations:
            coroutines.extend(transformation.generate_execution_plan(kg))
        return coroutines


async def run_coroutines(
    coroutines: t.List[t.Coroutine], desc: str, max_workers: int, callbacks: t.Optional[Callbacks] = None
):
    """
    Run a list of coroutines in parallel with optional callbacks.
    """
    for future in tqdm(
        await as_completed(coroutines, max_workers=max_workers),
        desc=desc,
        total=len(coroutines),
        leave=False,
    ):
        try:
            result = await future
            # Trigger callbacks if provided
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_llm_end') and result is not None:
                        callback.on_llm_end(result)
        except Exception as e:
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_llm_error'):
                        callback.on_llm_error(e)
            logger.error(f"Unable to apply transformation: {e}")


def get_desc(transform: BaseGraphTransformation | Parallel):
    if isinstance(transform, Parallel):
        transform_names = [t.__class__.__name__ for t in transform.transformations]
        return f"Applying [{', '.join(transform_names)}]"
    else:
        return f"Applying {transform.__class__.__name__}"


def apply_nest_asyncio():
    NEST_ASYNCIO_APPLIED: bool = False
    if is_event_loop_running():
        # an event loop is running so call nested_asyncio to fix this
        try:
            import nest_asyncio
        except ImportError:
            raise ImportError(
                "It seems like your running this in a jupyter-like environment. Please install nest_asyncio with `pip install nest_asyncio` to make it work."
            )

        if not NEST_ASYNCIO_APPLIED:
            nest_asyncio.apply()
            NEST_ASYNCIO_APPLIED = True


from ragas.cost import CostCallbackHandler, get_token_usage_for_openai


def apply_transforms(
    kg: KnowledgeGraph,
    transforms: Transforms,
    run_config: RunConfig = RunConfig(),
    callbacks: t.Optional[Callbacks] = None,
    cost_per_input_token: float = 0.01,
    cost_per_output_token: float = 0.02,
):
    """
    Apply a list of transformations to a knowledge graph in place and estimate the cost.
    """
    # Initialize callbacks if not provided
    callbacks = callbacks or []
    
    # Add a cost callback handler for tracking token usage
    cost_cb = CostCallbackHandler(token_usage_parser=get_token_usage_for_openai)
    
    # Add the cost callback to the callbacks list
    if isinstance(callbacks, list):
        callbacks.append(cost_cb)
    
    # Apply nest_asyncio to fix the event loop issue in Jupyter
    apply_nest_asyncio()
    
    # If single transformation, wrap it in a list
    if isinstance(transforms, BaseGraphTransformation):
        transforms = [transforms]
    
    # Apply the transformations
    if isinstance(transforms, t.List):
        for transform in transforms:
            # For LLM-based transforms, set callbacks if the transform has the attribute
            if hasattr(transform, 'llm') and transform.llm is not None:
                if hasattr(transform, 'callbacks'):
                    transform.callbacks = callbacks
                
                # If the transform's LLM has callback methods, set them too
                if hasattr(transform.llm, 'callbacks') and transform.llm.callbacks is None:
                    transform.llm.callbacks = callbacks
            
            # Execute the transform's plan
            asyncio.run(
                run_coroutines(
                    transform.generate_execution_plan(kg),
                    get_desc(transform),
                    run_config.max_workers,
                    callbacks=callbacks  # Pass callbacks to run_coroutines
                )
            )
    elif isinstance(transforms, Parallel):
        # For Parallel, wrap the callback setting in a try-except to handle various transform types
        for transform in transforms.transformations:
            if hasattr(transform, 'llm') and transform.llm is not None:
                if hasattr(transform, 'callbacks'):
                    transform.callbacks = callbacks
                
                # If the transform's LLM has callback methods, set them too
                if hasattr(transform.llm, 'callbacks') and transform.llm.callbacks is None:
                    transform.llm.callbacks = callbacks
                    
        asyncio.run(
            run_coroutines(
                transforms.generate_execution_plan(kg),
                get_desc(transforms),
                run_config.max_workers,
                callbacks=callbacks  # Pass callbacks here
            )
        )
    else:
        raise ValueError(
            f"Invalid transforms type: {type(transforms)}. Expects a list of BaseGraphTransformations or a Parallel instance."
        )
    
    # Calculate and return the total cost
    # Check if we have any token usage data
    if not hasattr(cost_cb, 'usage_data') or not cost_cb.usage_data:
        # If no usage data, return a default minimal cost or 0
        logger.warning("No token usage data was collected. Cost may be underestimated.")
        return 0.0
    
    try:
        total_cost = cost_cb.total_cost(
            cost_per_input_token=cost_per_input_token,
            cost_per_output_token=cost_per_output_token,
        )
        return total_cost
    except Exception as e:
        logger.error(f"Error calculating total cost: {e}")
        return 0.0
    
def rollback_transforms(kg: KnowledgeGraph, transforms: Transforms):
    """
    Rollback a list of transformations from a knowledge graph.

    Note
    ----
    This is not yet implemented. Please open an issue if you need this feature.
    """
    # this will allow you to roll back the transformations
    raise NotImplementedError