import numpy as np
from src.Model import Model
import pickle
from tqdm import tqdm
from simpful import FuzzySystem, AutoTriangle, LinguisticVariable


class FuzzyVariableManager:
    """Manages fuzzy variables (inputs and outputs) with custom N and range."""

    def __init__(self):
        self.variables = {}

    def create_variable(self, name: str, N:int, var_range:list, is_output: bool=False) -> LinguisticVariable:
        """
        Create fuzzy variable based on Auto Triangle and convert to Linguistic Variable.

        Args:
            name (str): Variable name
            N (int): Number of fuzzy sets (total sets = 2*N + 1)
            var_range (list): [min, max] universe of discourse for this variable
            is_output (bool): Whether variable is output
        """
        assert N > 0, "N must be greater than 0"
        assert len(var_range) == 2 and var_range[0] < var_range[1], \
            f"var_range must be [min, max] with min < max! VarName:{name}"

        regions = 2 * N + 1

        # region names (S=small, B=big, Z=zero)
        terms = []
        for i in range(regions):
            if i < N:
                terms.append(f"S{abs(i - N)}")
            elif i > N:
                terms.append(f"B{abs(i - N)}")
            else:
                terms.append("Z")

        auto = AutoTriangle(
            n_sets=regions,
            terms=terms,
            universe_of_discourse=var_range
        )

        lv = LinguisticVariable(auto._FSlist, concept=name, universe_of_discourse=var_range)
        self.variables[name] = {
            "lv": lv,
            "N": N,
            "range": var_range,
            "is_output": is_output
        }
        return lv

    def get(self, name:str) -> LinguisticVariable:
        return self.variables[name]["lv"]


class FuzzyRuleManager:
    """Manages a fuzzy rule base with adaptive learning and forgetting."""

    def __init__(self, prune_weight_threshold:float=0.1, prune_use_threshold:int=0,
                 prune_window:int=15, update_rule_window:int=15,max_rules:int=None, 
                 aggregation_fun="product", logger:...=None):

        self.rules = []
        self.weights = []
        self.usage_count = []
        self.error_contribution = []  # track per-rule contribution to error
        self.prune_count = 0

        self.prune_weight_threshold = prune_weight_threshold
        self.prune_use_threshold = prune_use_threshold
        self.prune_window = prune_window
        self.update_rule_window = update_rule_window
        self.max_rules = max_rules
        self.aggregation_fun = aggregation_fun

        self.logger = logger
        self.lv = 6

        assert self.prune_window > self.prune_use_threshold, \
            "prune_window must be greater than prune_use_threshold"
        
    @staticmethod
    def strong_pertinence(var:LinguisticVariable, value:float) -> tuple[str,float]:
        """Selects the term with the highest membership degree for a value."""
        values = var.get_values(value)
        terms = list(values.keys())
        vals = list(values.values())
        strong_term = terms[np.argmax(vals)]
        value_term = vals[np.argmax(vals)]
        return strong_term, value_term

    @staticmethod
    def build_rule(input_names: list[str], output_name: str, terms: str) -> str:
        """Creates rule string in simpful-compatible format."""
        rule_string = "IF "
        for i, name in enumerate(input_names):
            rule_string += f"({name} IS {terms[i]}) AND "
        rule_string = rule_string[:-4] + f"THEN ({output_name} IS {terms[-1]})"
        return rule_string

    def _aggregate_weights(self, weights:list[float]) -> float:
        """Aggregate membership degrees according to selected aggregation function."""
        if callable(self.aggregation_fun):
            return self.aggregation_fun(weights)
        elif self.aggregation_fun == "product":
            return np.prod(weights)
        elif self.aggregation_fun == "min":
            return min(weights)
        elif self.aggregation_fun == "max":
            return max(weights)
        elif self.aggregation_fun == "arit_mean":
            return np.mean(weights)
        else:
            raise Exception(f"Unknown aggregation function {self.aggregation_fun}")

    def update_rules(self, input_vars: list[LinguisticVariable], output_var: LinguisticVariable,
                      values_io:list[float], var_names:list[str]) -> None:
        """Updates rule base (learns new rules or updates existing ones)."""
        terms_list, weight_list = [], []

        fuzzy_vars = input_vars + [output_var]
        for i in range(len(fuzzy_vars)):
            term, weight = self.strong_pertinence(fuzzy_vars[i], values_io[i])
            terms_list.append(term)
            weight_list.append(weight)

        new_weight = self._aggregate_weights(weight_list)
        new_rule = self.build_rule(var_names[:-1], var_names[-1], terms_list)

        if not any(new_rule[:new_rule.find('THEN')] in item for item in self.rules) or len(self.rules) == 0:
            # Check maximum rules limit
            if self.max_rules is not None and len(self.rules) >= self.max_rules:
                #tqdm.write(f"Replacing rule {idx}: old_weight = {self.weights[idx]} -> new_weight = {new_weight}.")
                # Replace weakest rule
                idx = np.argmin(self.weights)
                
                if self.logger:
                    self.logger.log(self.lv, f"Replacing rule {idx}: old_weight = {self.weights[idx]} -> new_weight = {new_weight}.")
                
                self.rules[idx] = new_rule
                self.weights[idx] = new_weight
                self.usage_count[idx] = 0
                self.error_contribution[idx] = 0.0
            else:
                #tqdm.write(f"Add new rule: now the fuzzy uses {len(self.rules)+1} rules.", end='\r')
                if self.logger:
                    self.logger.log(self.lv, f"Add new rule: now the fuzzy uses {len(self.rules)+1} rules.", end='\r')
                self.rules.append(new_rule)
                self.weights.append(new_weight)
                self.usage_count.append(0)
                self.error_contribution.append(0.0)
        else:
            arr = np.array(self.rules)
            mask = np.core.defchararray.find(arr.astype(str), new_rule[:new_rule.find('THEN')])
            old_weight = self.weights[mask[0]]
            if new_weight > old_weight:
                self.rules[mask[0]] = new_rule
                self.weights[mask[0]] = new_weight
    
    def register_rule_usage(self, prediction_error:float=None) -> None:
        """
        Increment usage count for rules with weights above threshold.
        Optionally register contribution to global error.new_rule
        """
        for idx, weight in enumerate(self.weights):
            if weight > self.prune_weight_threshold:
                self.usage_count[idx] += 1
                if prediction_error is not None:
                    self.error_contribution[idx] += abs(prediction_error)

    def prune_unused_rules(self) -> bool:
        """
        Remove unused or low-impact rules based on sliding window.
        Returns True if pruning occurred.
        """
        if self.prune_count >= self.prune_window:
            to_remove = []
            for i, count in enumerate(self.usage_count):
                avg_error = self.error_contribution[i] / max(1, count)
                if count <= self.prune_use_threshold and avg_error < self.prune_weight_threshold:
                    to_remove.append(i)

            if len(to_remove) > 0:
                #tqdm.write(f"Pruning {len(to_remove)} rules out of {len(self.rules)} "
                #            f"-> {len(self.rules) - len(to_remove)} remaining.")
                self.log(f"Pruning {len(to_remove)} rules out of {len(self.rules)} "
                            f"-> {len(self.rules) - len(to_remove)} remaining.")
                for idx in sorted(to_remove, reverse=True):
                    del self.rules[idx]
                    del self.weights[idx]
                    del self.usage_count[idx]
                    del self.error_contribution[idx]

            # Reset usage counters after pruning
            self.usage_count = [0 for _ in self.rules]
            self.error_contribution = [0.0 for _ in self.rules]
            self.prune_count = 0
            return True
        else:
            self.prune_count += 1
            return False


class FuzzyTSModel(Model):
    """Fuzzy model for time series forecasting with adaptive rule management."""

    def __init__(self, input_configs:list[dict], output_config:dict,
                 update_rule_window:int=15,max_rules:int=None, aggregation_fun="product"):
        """
        Args:
            input_configs (list of dict): [{"name": str, "N": int, "range": [min,max]}, ...]
            output_config (dict): {"name": str, "N": int, "range": [min,max]}
        """
        super().__init__()

        # it'll be useful for loading process
        self.input_configs = input_configs
        self.output_config = output_config
        self.update_rule_window = update_rule_window
        self.max_rules = max_rules
        self.aggregation_fun = aggregation_fun


        self.input_names = [cfg["name"] for cfg in input_configs]
        self.output_name = output_config["name"]
        self.var_manager = FuzzyVariableManager()

        self.rule_manager = FuzzyRuleManager(max_rules=max_rules, update_rule_window=update_rule_window,
                                             aggregation_fun=aggregation_fun, logger=self.logger)
        self.fs = FuzzySystem(show_banner=False)

        # Create variables
        self.input_vars = [
            self.var_manager.create_variable(cfg["name"], cfg["N"], cfg["range"])
            for cfg in input_configs
        ]
        self.output_var = self.var_manager.create_variable(
            output_config["name"], output_config["N"], output_config["range"], is_output=True
        )

        # Add variables to fuzzy system
        self.fs.add_linguistic_variable(self.output_name, self.output_var)
        for name, var in zip(self.input_names, self.input_vars):
            self.fs.add_linguistic_variable(name, var)
        
        # Useful for adaptive pruning
        self.X_train_dim = None

        # Return some logs
        self.log("FuzzyTSModel initialized.")
        self.log(f"Input variables: {self.input_names}")
        self.log(f"Output variable: {self.output_name}")
        self.log(f"Number of sets (N), range per variable:")
        for cfg in input_configs:
            self.log(f"{cfg['name']}: N={cfg['N']}, range={cfg['range']}")

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Batch learning of fuzzy rules from dataset."""
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        assert X.shape[0] == y.shape[0], "X and y must have same first dimension"
        self.X_train_dim = X.shape
        
        self.log(f"Starting training with {X.shape[0]} samples and {X.shape[1]} input variables.")

        for xi, yi in tqdm(zip(X, y), total=X.shape[0], desc="Fitting model"):
            values_io = list(xi) + [yi]
            self.rule_manager.update_rules(self.input_vars, self.output_var, values_io,
                                           self.input_names + [self.output_name])
        self.fs.add_rules(self.rule_manager.rules)
        self.is_fitted = 1

        self.log(f"Training completed. Learned {len(self.rule_manager.rules)} rules.")

    def partial_fit(self, xi:np.ndarray, yi:float) -> None:
        """
        Online update with a single sample (incremental learning).
        Args:
            xi (np.ndarray): Input vector
            yi (float): Target value
        """
        values_io = list(xi) + [yi]
        self.rule_manager.update_rules(self.input_vars, self.output_var, values_io,
                                       self.input_names + [self.output_name])
        self.fs._rules.clear()
        self.fs.add_rules(self.rule_manager.rules)
        self.is_fitted = 1

    def predict(self, X:np.ndarray, y_true:np.ndarray=None) -> np.ndarray:
        """
        Predict output for given input data.
        Optionally receives y_true to track error contribution for pruning.
        """

        if not self.is_fitted:
            self.log("Prediction aborted: model is not fitted.", level="WARNING")
            raise RuntimeError("Model must be fitted before prediction.")
        
        X = np.asarray(X)
        assert X.shape[1:] == self.X_train_dim[1:], "Input shape mismatch"
        
        self.log(f"Starting prediction for {X.shape[0]} samples.")
        
        predictions = []
        for idx, xi in enumerate(tqdm(X, total=X.shape[0], desc="Predicting")):
            for name, val in zip(self.input_names, xi):
                self.fs.set_variable(name, val)
            result = self.fs.inference()
            
            y_pred = result[self.output_name]
            predictions.append(y_pred)

            # Rule usage + adaptive pruning
            error = None
            if y_true is not None:
                error = y_true[idx] - y_pred

            self.rule_manager.register_rule_usage(prediction_error=error)
            if self.rule_manager.prune_unused_rules():
                assert len(self.rule_manager.rules) > 0, "All rules were pruned. Adjust pruning parameters."
                self.fs._rules.clear()
                self.fs.add_rules(self.rule_manager.rules)
        
        self.log("Prediction completed.")
        return np.array(predictions)

    def explain(self) -> list[str]:
        """Return current rule base as list of strings."""
        rules = self.rule_manager.rules
        if not rules:
            self.log("No rules available to explain.", level="WARNING")
            return ["No rules learned."]
        self.log(f"Explaining {len(rules)} learned rules.")
        return [str(r) for r in rules]

    def predict_and_update(self, X: np.ndarray, y_true: np.ndarray=None,
                       abs_error_threshold: float=0.05, rel_error_threshold: float=None,
                       verbose: bool=True) -> np.ndarray:
        """
        Predict and optionally update rules online if prediction error exceeds thresholds.
        This version uses small batch prediction for efficiency.

        Args:
            X (np.ndarray): Input samples, shape (n_samples, n_features).
            y_true (np.ndarray): Ground truth values, shape (n_samples,).
            abs_error_threshold (float): Absolute error threshold for update.
            rel_error_threshold (float): Relative error threshold (percentage). Optional.
            verbose (bool): Whether to print updates when model learns online.
        """
        # Standard batch prediction
        update_rule_window = self.rule_manager.update_rule_window if y_true is not None else X.shape[0] # no updates if y_true not provided
        y_pred = np.zeros(X.shape[0]) # output vector
        
        
        for start in range(0, X.shape[0], update_rule_window):
            end = min(start + update_rule_window, X.shape[0])
            X_batch = X[start:end]
            y_batch_true = y_true[start:end] if y_true is not None else None
            y_batch_pred = self.predict(X_batch, y_true=y_batch_true)

            # Compute errors
            abs_errors = np.abs(y_batch_true - y_batch_pred) if y_true is not None else None

            # Relative error only if requested
            rel_errors = None
            if rel_error_threshold is not None:
                rel_errors = np.zeros_like(abs_errors)
                nonzero_mask = np.abs(y_batch_true) > 1e-8 # avoid division by zero
                rel_errors[nonzero_mask] = abs_errors[nonzero_mask] / np.abs(y_batch_true[nonzero_mask])

            # Decide which samples trigger update
            for xi, yi, yp, err in zip(X_batch, y_batch_true, y_batch_pred, abs_errors):
                update = False
                if abs_error_threshold is not None and err > abs_error_threshold:
                    update = True
                if rel_error_threshold is not None and rel_errors is not None:
                    idx = np.where(y_true == yi)[0][0]  # index of current sample
                    if rel_errors[idx] > rel_error_threshold:
                        update = True

                if update:
                    if verbose:
                        tqdm.write(f"Updating model: y_true={yi}, y_pred={yp:.4f}, abs_err={err:.4f}")
                    self.partial_fit(xi, yi)

            y_pred[start:end] = y_batch_pred

        return y_pred

    def save(self, filepath: str):
        """Save model to file (pickle)."""
        state = {
            "input_configs": self.input_configs,
            "output_config": self.output_config,
            "update_rule_window": self.update_rule_window,
            "max_rules": self.max_rules,
            "aggregation_fun": self.aggregation_fun,
            "rules": self.rule_manager.rules,
            "is_fitted": self.is_fitted,
            "X_train_dim": self.X_train_dim
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        self.log(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, **kwargs):
        """Load model from file and rebuild."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        # recreate the model with stored params
        model = cls(
            input_configs=state["input_configs"],
            output_config=state["output_config"],
            update_rule_window=state["update_rule_window"],
            max_rules=state["max_rules"],
            aggregation_fun=state["aggregation_fun"],
            **kwargs
        )

        # restore rules
        model.rule_manager.rules = state["rules"]
        model.fs.add_rules(model.rule_manager.rules)
        model.is_fitted = state["is_fitted"]
        model.X_train_dim = state["X_train_dim"]

        model.log(f"Model loaded from {filepath} with {len(model.rule_manager.rules)} rules.")
        return model
