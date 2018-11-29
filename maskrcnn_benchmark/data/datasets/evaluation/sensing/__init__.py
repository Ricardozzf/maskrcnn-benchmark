import logging

from .sensing_eval import do_sensing_evaluation


def voc_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("sensing evaluation doesn't support box_only, ignored.")
    logger.info("performing sensing evaluation, ignored iou_types.")
    return do_sensing_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
