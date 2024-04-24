"""Implementation of the RMIA attack."""
import numpy as np
import torch
import torch.nn.functional as F


from typing import Optional, List

from leakpro.dataset import get_dataset_subset
from leakpro.import_helper import Self
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.mia_attacks.attack_utils import AttackUtils
from leakpro.mia_attacks.attacks.attack import AttackAbstract
from leakpro.signals.signal import ModelLogits


class AttackRMIA(AttackAbstract):
    """Implementation of the RMIA attack."""

    def __init__(self:Self, attack_utils: AttackUtils, configs: dict) -> None:
        """Initialize the RMIA attack.

        Args:
        ----
            attack_utils (AttackUtils): Utility class for the attack.
            configs (dict): Configuration parameters for the attack.

        """
        # Initializes the parent metric
        super().__init__(attack_utils)

        self.shadow_models = attack_utils.attack_objects.shadow_models
        self.data_module = attack_utils.data_module
        self.method = "offline" # TODO: add online and make this a config parameter
        # TODO: adapt a and b for different datasets
        self.offline_a = 1 # parameter from which we compute p(x) from p_OUT(x) such that p_IN(x) = a p_OUT(x) + b.
        self.offline_b: 0
        self.gamma = 2.0 # threshold for the attack
        self.temperature = 2.0 # temperature for the softmax

        # self.f_attack_data_size = configs["audit"].get("f_attack_data_size", 0.3)

        # self.signal = ModelLogits()
        # self.epsilon = 1e-6
        self.ratio_z: Optional[float] = None

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "RMIA attack"
        reference_str = "Zarifzadeh, Sajjad, Philippe Cheng-Jie Marc Liu, and Reza Shokri. \
            Low-Cost High-Power Membership Inference by Boosting Relativity. (2023)."
        summary_str = "The RMIA attack is a membership inference attack based on the output logits of a black-box model."
        detailed_str = "The attack is executed according to: \
            1. A fraction of the population is sampled to compute the likelihood LR_z of p(z|theta) to p(z) for the target model.\
            2. The ratio is used to compute the likelihood ratio LR_x of p(x|theta) to p(x) for the target model. \
            3. The ratio LL_x/LL_z is viewed as a random variable (z is random) and used to classify in-members and out-members. \
            4. The attack is evaluated on an audit dataset to determine the attack performance."
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }


    # def softmax(self:Self, all_logits:np.ndarray,
    #             true_label_indices:np.ndarray,
    #             return_full_distribution:bool=False) -> np.ndarray:
    #     """Compute the softmax function.

    #     Args:
    #     ----
    #         all_logits (np.ndarray): Logits for each class.
    #         true_label_indices (np.ndarray): Indices of the true labels.
    #         return_full_distribution (bool, optional): return the full distribution or just the true class probabilities.

    #     Returns:
    #     -------
    #         np.ndarray: Softmax output.

    #     """
    #     logit_signals = all_logits / self.temperature
    #     max_logit_signals = np.max(logit_signals,axis=2)
    #     logit_signals = logit_signals - max_logit_signals.reshape(1,-1,1)
    #     exp_logit_signals = np.exp(logit_signals)
    #     exp_logit_sum = np.sum(exp_logit_signals, axis=2)

    #     if return_full_distribution is False:
    #         true_exp_logit =  exp_logit_signals[:, np.arange(exp_logit_signals.shape[1]), true_label_indices]
    #         output_signal = true_exp_logit / exp_logit_sum
    #     else:
    #         output_signal = exp_logit_signals / exp_logit_sum[:,:,np.newaxis]
    #     return output_signal

    @staticmethod
    def get_probability_from_model_output(logit: torch.Tensor, label: torch.Tensor)-> float:
        """Calculates P(y_predicted=y_true) from the model output (logits) and the label. Designed for binary classification with a single output node.

        Args:
            logit (torch.Tensor): Model output before sigmoid.
            label (torch.Tensor): Correct label.

        Returns:
            float: Predicted probability of the true label.
        """
        logit = logit.squeeze()
        label = label.squeeze()
        assert logit.dim() == label.dim(), "In get_probability_from_sigmoid_output the logit and label tensors have different dimensions"
        if logit.dim() == 0:
            y = label.item()
            y_head = F.sigmoid(logit).item()
            probability_for_true_label = y_head if y ==1 else (1-y_head)
        else:
            raise NotImplementedError
        return probability_for_true_label

    def prepare_attack(self:Self) -> None:
        """Prepare data needed for running the attack on the target model and dataset.

        Signals are computed on the auxiliary model(s) and dataset.
        """
        # prepare data
        # TODO: Optimize by bigger batch size
        # TODO: Discuss which datasets makes sense to include here (for now ALL data is included)
        population_dataloader =  self.data_module.population_dataloader()

        p_z_given_different_thetas_list: List[List[float]] = []

        # iterate over population
        for z, y in iter(population_dataloader):
            # iterate over all the shadow models
            temp = []
            for shadow_model in self.shadow_models:
                with torch.no_grad():
                    if isinstance(z, torch.Tensor):
                        logit = shadow_model.forward(z.to(shadow_model.device))
                    else:
                        logit = shadow_model.forward(z)
                    p_z_shadow_model = self.get_probability_from_model_output(logit, y)
                    temp.append(p_z_shadow_model)
            p_z_given_different_thetas_list.append(temp)


        # get P(z) by taking the average of it (casting float to make type checker happy)
        p_z = np.mean(np.array(p_z_given_different_thetas_list), axis=1)


        # very weird probability shift here. TODO: See if it is more realistic after actually determening a and b
        if self.method == "offline":
            p_z = 0.5*((self.offline_a + 1) * p_z + (1-self.offline_a))

        # get P(z|theta_target)
        p_z_target_model_list: List[float] = []

        for z, y in iter(population_dataloader):
            with torch.no_grad():
                if isinstance(z, torch.Tensor):
                    logit = self.target_model.forward(z.to(self.target_model.device))
                else:
                    logit = self.target_model.forward(z)
                p_z_target_model = self.get_probability_from_model_output(logit, y)
                p_z_target_model_list.append(p_z_target_model)

        p_z_given_theta = np.array(p_z_target_model_list)
        self.ratio_z = p_z_given_theta / p_z 

        # # sample dataset to compute histogram
        # all_index = np.arange(self.population_size)
        # attack_data_size = np.round(
        #     self.f_attack_data_size * self.population_size
        # ).astype(int)

        # self.attack_data_index = np.random.choice(
        #     all_index, attack_data_size, replace=False
        # )
        # attack_data = get_dataset_subset(self.population, self.attack_data_index)

        # # compute the ratio of p(z|theta) (target model) to p(z)=sum_{theta'} p(z|theta') (shadow models)
        # # for all points in the attack dataset output from signal: # models x # data points x # classes

        # # get the true label indices
        # z_label_indices = np.array(attack_data.y)

        # # run points through real model to collect the logits
        # logits_theta = np.array(self.signal([self.target_model], attack_data))
        # # collect the softmax output of the correct class
        # p_z_given_theta = self.softmax(logits_theta, z_label_indices)

        # # run points through shadow models and collect the logits
        # logits_shadow_models = self.signal(self.shadow_models, attack_data)
        # # collect the softmax output of the correct class for each shadow model
        # p_z_given_shadow_models = [self.softmax(np.array(x).reshape(1,*x.shape), z_label_indices) for x in logits_shadow_models]
        # # stack the softmax output of the correct class for each shadow model to dimension # models x # data points
        # p_z_given_shadow_models = np.array(p_z_given_shadow_models).squeeze()

        # # evaluate the marginal p(z)
        # p_z = np.mean(p_z_given_shadow_models, axis=0) if len(self.shadow_models) > 1 else p_z_given_shadow_models.squeeze()
        # p_z = 0.5*((self.offline_a + 1) * p_z + (1-self.offline_a))

        # #TODO: pick the maximum value of the softmax output in p(z)
        # self.ratio_z = p_z_given_theta / (p_z + self.epsilon)


    def run_attack(self:Self) -> CombinedMetricResult:
        """Run the attack on the target model and dataset.

        Args:
        ----
            fpr_tolerance_rate_list (optional): List of FPR tolerance values that may be used by the threshold function
                to compute the attack threshold for the metric.

        Returns:
        -------
            Result(s) of the metric.

        """

        self.data_module.usecase = "target"
        self.data_module.setup(stage="fit")
        member_dataloader = self.data_module.train_dataloader()
        self.data_module.setup(stage="test")
        no_member_dataloader = self.data_module.test_dataloader()

        # get the logits for the audit dataset
        # shadow models
        # we need to have a list of list here since we don't average over all datapoints like for p(z)
        p_x_given_different_thetas_list_members: List[List[float]] = []
        p_x_given_different_thetas_list_no_members: List[List[float]] = []
        # iterate over members
        for x, y in iter(member_dataloader):
            # iterate over all the shadow models
            temp = []
            for shadow_model in self.shadow_models:
                with torch.no_grad():
                    if isinstance(x, torch.Tensor):
                        logit = shadow_model.forward(x.to(shadow_model.device))
                    else:
                        logit = shadow_model.forward(x)
                    p_x_shadow_model = self.get_probability_from_model_output(logit, y)
                    temp.append(p_x_shadow_model)
            p_x_given_different_thetas_list_members.append(temp)

        for x, y in iter(no_member_dataloader):
            # iterate over all the shadow models
            temp = []
            for shadow_model in self.shadow_models:
                with torch.no_grad():
                    if isinstance(x, torch.Tensor):                
                        logit = shadow_model.forward(x.to(shadow_model.device))
                    else:
                        logit = shadow_model.forward(x)
                    p_x_shadow_model = self.get_probability_from_model_output(logit, y)
                    temp.append(p_x_shadow_model)
            p_x_given_different_thetas_list_no_members.append(temp)

        # target model
        p_x_given_target_list_members: List[float] = []
        p_x_given_target_list_no_members: List[float] = []
        for x, y in iter(member_dataloader):
            with torch.no_grad():
                if isinstance(x, torch.Tensor):
                    logit = self.target_model.forward(x.to(self.target_model.device))
                else:
                    logit = self.target_model.forward(x)
                p_x_target_model = self.get_probability_from_model_output(logit, y)
                p_x_given_target_list_members.append(p_x_target_model)
        for x, y in iter(no_member_dataloader):
            with torch.no_grad():
                if isinstance(x, torch.Tensor):
                    logit = self.target_model.forward(x.to(self.target_model.device))
                else:
                    logit = self.target_model.forward(x)
                p_x_target_model = self.get_probability_from_model_output(logit, y)
                p_x_given_target_list_no_members.append(p_x_target_model)


        # make lists numpy arrays for easier computation
        p_x_given_different_thetas_array_members = np.array(p_x_given_different_thetas_list_members)
        p_x_given_different_thetas_array_no_members = np.array(p_x_given_different_thetas_list_no_members)
        p_x_given_target_members = np.array(p_x_given_target_list_members)
        p_x_given_target_no_members = np.array(p_x_given_target_list_no_members)

        # calculate p_x for each datapoint by averaging over the different shadow models    
        p_x_members = np.mean(p_x_given_different_thetas_array_members, axis=1)
        if self.method == "offline":
            p_x_members = 0.5*((self.offline_a + 1) * p_x_members + (1-self.offline_a))
        p_x_no_members = np.mean(p_x_given_different_thetas_array_no_members, axis=1)
        if self.method == "offline":
                    p_x_no_members = 0.5*((self.offline_a + 1) * p_x_no_members + (1-self.offline_a))

        # compute ratios for x
        ratio_x_members = p_x_given_target_members / p_x_members # (X,)
        ratio_x_no_members = p_x_given_target_no_members / p_x_no_members

        # compute scores
        # members
        ratio_x_members_reshaped = ratio_x_members[:, np.newaxis] # (X,1)
        likelihood_members = ratio_x_members_reshaped / self.ratio_z # (X,Z) will be broadcasted
        score_members = np.mean(likelihood_members > self.gamma, axis=1) # (X,)
        # no members
        ratio_x_no_members_reshaped = ratio_x_no_members[:, np.newaxis] # (X,1)
        likelihood_no_members = ratio_x_no_members_reshaped / self.ratio_z # (X,Z) will be broadcasted
        score_no_members = np.mean(likelihood_no_members > self.gamma, axis=1) # (X,)

        # Creating thresholds for comparison
        thresholds = np.linspace(1/likelihood_members.shape[1], 1, 1000)

        # Predicting membership
        member_preds = np.greater(score_members[:, np.newaxis], thresholds).T
        non_member_preds = np.greater(score_no_members[:, np.newaxis], thresholds).T

        # Concatenating predictions and setting true labels
        predictions = np.concatenate([member_preds, non_member_preds], axis=1)
        true_labels = np.concatenate(
            [np.ones(len(score_members)), np.zeros(len(score_no_members))]
        )
        signal_values = np.concatenate(
            [score_members, score_no_members]
        )

        # Compute ROC, TP, TN, etc.
        return CombinedMetricResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values=signal_values,
        )

        # # get the logits for the audit dataset
        # audit_data = get_dataset_subset(self.population, self.audit_dataset["data"])
        # x_label_indices = np.array(audit_data.y)

        # # run target points through real model to get logits
        # logits_theta = np.array(self.signal([self.target_model], audit_data))
        # # collect the softmax output of the correct class
        # p_x_given_theta = self.softmax(logits_theta, x_label_indices)

        # # run points through shadow models and collect the logits
        # logits_shadow_models = self.signal(self.shadow_models, audit_data)
        # # collect the softmax output of the correct class for each shadow model
        # p_x_given_shadow_models = [self.softmax(np.array(x).reshape(1,*x.shape), x_label_indices) for x in logits_shadow_models]
        # # stack the softmax output of the correct class for each shadow model
        # # to dimension # models x # data points
        # p_x_given_shadow_models = np.array(p_x_given_shadow_models).squeeze()
        # # evaluate the marginal p_out(x) by averaging the output of the shadow models
        # p_x_out = np.mean(p_x_given_shadow_models, axis=0) if len(self.shadow_models) > 1 else p_x_given_shadow_models.squeeze()

        # # compute the marginal p(x) from P_out and p_in where p_in = a*p_out+b
        # p_x = 0.5*((self.offline_a + 1) * p_x_out + (1-self.offline_a))

        # # compute the ratio of p(x|theta) to p(x)
        # ratio_x = p_x_given_theta / (p_x + self.epsilon)

        # # for each x, compare it with the ratio of all z points
        # likelihoods = ratio_x.T / self.ratio_z
        # score = np.mean(likelihoods > self.gamma, axis=1)

        # # pick out the in-members and out-members signals
        # self.in_member_signals = score[self.audit_dataset["in_members"]].reshape(-1,1)
        # self.out_member_signals = score[self.audit_dataset["out_members"]].reshape(-1,1)

        # thresholds = np.linspace(1/likelihoods.shape[1], 1, 1000)


        # member_preds = np.greater(self.in_member_signals, thresholds).T
        # non_member_preds = np.greater(self.out_member_signals, thresholds).T

        # # what does the attack predict on test and train dataset
        # predictions = np.concatenate([member_preds, non_member_preds], axis=1)
        # # set true labels for being in the training dataset
        # true_labels = np.concatenate(
        #     [
        #         np.ones(len(self.in_member_signals)),
        #         np.zeros(len(self.out_member_signals)),
        #     ]
        # )
        # signal_values = np.concatenate(
        #     [self.in_member_signals, self.out_member_signals]
        # )

        # # compute ROC, TP, TN etc
        # return CombinedMetricResult(
        #     predicted_labels=predictions,
        #     true_labels=true_labels,
        #     predictions_proba=None,
        #     signal_values=signal_values,
        # )


