import logging
from polyaxon_client.tracking import Experiment, get_log_level, get_outputs_path
import argparse

from src.modelling.model import Model

logger = logging.getLogger(__name__)


def run_experiment(data_path, model_name, params):
    try:
        log_level = get_log_level()
        if not log_level:
            log_level = logging.INFO

        logger.info("Starting experiment")

        experiment = Experiment()
        logging.basicConfig(level=log_level)

        # initiate model class
        model = Model(model_name)
        logger.info(f'{model_name} ok')

        # get data
        refs = model.get_data(data_path, **params)
        logger.info('data ok')

        # train model
        model.model.train()
        logger.info('model trained')

        # get pred
        preds = refs.apply(lambda x: model.model.predict(x))
        logger.info('preds ok')

        # eval
        precision, recall, f1 = model.model.eval(preds, refs)
        logger.info('eval ok')

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')

        experiment.log_metrics(precision=precision)
        experiment.log_metrics(recall=recall)
        experiment.log_metrics(f1=f1)

        logger.info("Experiment completed")
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--model_name', default='ModelHF')
    parser.add_argument('--samples', default=100)
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()

    data_path = args.data_path
    model_name = args.model_name

    params = {}

    params = {
            'samples': int(args.samples),
            'seed': int(args.seed)
            }

    print('params used:', params)
    
    run_experiment(data_path, model_name, params)