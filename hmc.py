"""
The `hmc` module is a decision tree based model for hierachical multi-classification.
Adapted from https://github.com/davidwarshaw/hmc
"""


import warnings
from time import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from exceptions import *

__all__ = ["ClassHierarchy", "HierarchicalMultilabelClassifier"]

# =============================================================================
# Class Hierarchy
# =============================================================================


class ClassHierarchy:
    """
    Class for class heirarchy.

    Parameters
    ----------
        root :

    Attributes
    ----------

    """

    def __init__(self, root):
        self.root = root
        self.nodes = {}

    def _get_parent(self, child):
        # Return the parent of this node
        return (
            self.nodes[child]
            if (child in self.nodes and child != self.root)
            else self.root
        )

    def _get_children(self, parent):
        # Return a list of children nodes in alpha order
        return sorted(
            [
                child
                for child, childs_parent in self.nodes.items()
                if childs_parent == parent
            ]
        )

    def _get_ancestors(self, child):
        # Return a list of the ancestors of this node
        # Not including root, not including the child
        ancestors = []
        while True:
            child = self._get_parent(child)
            if child == self.root:
                break
            ancestors.append(child)
        return ancestors

    def _get_ancestors_of(self, children):
        # Return a list of the ancestors of list of nodes
        # Not including root, not including the child
        ancestors = []
        for child in children:
            ancestors_ = []
            while True:
                child = self._get_parent(child)
                if child == self.root:
                    break
                ancestors_.append(child)
            if len(ancestors_) > 0:
                ancestors.extend(ancestors_)
        return list(set(ancestors))

    def _get_descendants(self, parent):
        # Return a list of the descendants of this node
        # Not including the parent
        descendants = []
        self._depth_first(parent, descendants)
        descendants.remove(parent)
        return descendants

    def _get_descendants_of(self, nodes):
        # Return a list of the descendants of a list of nodes
        # Not including the parent
        all_descendants = []
        for parent in nodes:
            descendants = []
            self._depth_first(parent, descendants)
            descendants.remove(parent)
            all_descendants.extend(descendants)
        return all_descendants

    def _get_descendants_ascendants_of(self, nodes):
        # Return a list of the nodes in all branches
        # of a given list of nodes, not including the parent
        all_branch_nodes = []
        for parent in nodes:
            # towards leaf
            descendants = []
            self._depth_first(parent, descendants)
            descendants.remove(parent)

            # towards root
            ancestors = []
            while True:
                parent = self._get_parent(parent)
                if parent == self.root:
                    break
                ancestors.append(parent)

            if len(ancestors) > 0:
                all_branch_nodes.extend(ancestors)
            if len(descendants) > 0:
                all_branch_nodes.extend(descendants)

        return list(set(all_branch_nodes))

    def _is_descendant(self, parent, child):
        while child != self.class_hierarchy.root and child != parent:
            child = self.class_hierarchy._get_parent(child)
        return child == parent

    def _is_ancestor(self, parent, child):
        return _is_descendant(child, parent)

    def _depth_first_print(self, parent, indent, last):
        print(indent, end="")
        if last:
            print("\u2514\u2500", end="")
            indent += "  "
        else:
            print("\u251C\u2500", end="")
            indent += "\u2502 "
        print(parent)
        num_nodes = len(self._get_children(parent))
        node_count = 0
        for node in self._get_children(parent):
            node_count += 1
            self._depth_first_print(node, indent, node_count == num_nodes)

    def _depth_first(self, parent, classes):
        classes.append(parent)
        for node in self._get_children(parent):
            self._depth_first(node, classes)

    def add_node(self, child, parent):
        """
        Add a child-parent node to the class hierarchy.
        """
        if child == self.root:
            raise ValueError(
                "The hierarchy root: " + child + " is not a valid child node."
            )
        if child in self.nodes.keys():
            if self.nodes[child] != parent:

                print(
                    "Node: "
                    + str(child)
                    + ": "
                    + parent
                    + " has already been assigned parent: "
                    + self.nodes[child]
                )
                raise ValueError(
                    "Node: "
                    + str(child)
                    + " has already been assigned parent: "
                    + str(child)
                )
            else:
                return
        self.nodes[child] = parent

    def nodes_(self):
        """
        Return the hierarchy classes as a list of child-parent nodes.
        """
        return self.nodes

    def classes_(self):
        """
        Return the hierarchy classes as a list of unique classes.
        """
        classes = []
        self._depth_first(self.root, classes)
        return classes

    def print_(self):
        """
        Pretty print the class hierarchy.
        """
        self._depth_first_print(self.root, "", True)


# =============================================================================
# Decision Tree Hierarchical Classifier
# =============================================================================


class HierarchicalMultilabelClassifier:
    def __init__(self, class_hierarchy, classifier="lr", features_format="tfidf"):
        self.classifier = classifier
        self.features_format = features_format
        self.stages = []
        print("Initialising HMC...")
        time_start = time()
        self.class_hierarchy = class_hierarchy
        self._depth_first_stages(self.stages, self.class_hierarchy.root, 0)
        print(f"Initialised, in {time()-time_start} seconds")

    def _depth_first_class_prob(self, tree, node, indent, last, hand):
        if node == -1:
            return
        print(indent, end="")
        if last:
            print("\u2514\u2500", end="")
            indent += "    "
        else:
            print("\u251C\u2500", end="")
            indent += "\u2502   "
        print(hand + " " + node.encode("utf-8"))
        for k, count in enumerate(tree.tree_.value[node][0]):
            print(
                indent
                + tree.classes_[k].encode("utf-8")
                + ":"
                + stage(count / tree.tree_.n_node_samples[node], 2).encode("utf-8")
            )
        self._depth_first_class_prob(
            tree, tree.tree_.children_right[node], indent, False, "R"
        )
        self._depth_first_class_prob(
            tree, tree.tree_.children_left[node], indent, True, "L"
        )

    def _depth_first_stages(self, stages, parent, depth):
        # Get the children of this parent
        children = self.class_hierarchy._get_children(parent)
        # If there are children, build a classification stage

        ancestors = self.class_hierarchy._get_ancestors(parent)
        ancestor = None
        if parent != self.class_hierarchy.root:
            stage = {}
            stage["depth"] = depth
            stage["stage"] = parent
            stage["tree"] = []
            stage["labels"] = children
            stage["target"] = "target_stage_" + parent
            # change relabel target classes
            stage["class"] = stage["stage"]
            if len(ancestors) > 0:
                ancestor = ancestors[0]
            else:
                ancestor = self.class_hierarchy.root
            stage["ancestor"] = ancestor

            stages.append(stage)
        # Recurse through children
        for node in children:
            self._depth_first_stages(stages, node, depth + 1)

    def _recode_labels(self, stage, labels):
        # Reassign labels to their parents until either we 
        # hit the root, or an output class
        output = set()

        for label in labels:
            while label != self.class_hierarchy.root and label not in stage["class"]:
                label = self.class_hierarchy._get_parent(label)
            output.add(label)
        if stage["class"] in output:
            return stage["class"]
        else:
            return stage["ancestor"]

    def _prep_data(self, X, y):
        # Design matrix columns
        dm_cols = range(0, X.shape[1])
        # Target columns
        target = X.shape[1]
        print("X: ", X.shape)

        if self.features_format == "tfidf":
            print("y: ", y.shape)
            df = pd.concat([X, y], axis=1, ignore_index=True)
        elif self.features_format == "df":
            print("y: ", len(y))
            df = pd.concat(
                [pd.DataFrame(X), pd.DataFrame({"topics": y})],
                axis=1,
                ignore_index=True,
            )

        # Create a target column for each stage with the recoded labels
        for stage_number, stage in enumerate(self.stages):
            df[stage["target"]] = pd.DataFrame.apply(
                df[[target]],
                lambda row: self._recode_labels(stage, row[target]),
                axis=1,
            )
        return df, dm_cols

    def fit(self, X, y):
        """
        Build a decision tree multi-classifier from training data (X, y).
        """
        # Prep data
        print(f"Preparing data...")
        time_start = time()
        df, dm_cols = self._prep_data(X, y)
        print(f"Data prepared, {time()-time_start} seconds")

        # Fit each stage
        for stage_number, stage in enumerate(self.stages):
            print("=" * 50)
            print(f"Fitting stage {str(stage_number)}, {stage['target']} - START")

            # filtering the data for this stage
            selection = [stage["class"], stage["ancestor"]]
            mask = df[stage["target"]].apply(
                lambda x: any(item for item in selection if item in x)
            )
            df1 = df[mask]

            dm = df1[dm_cols]
            y_stage = df[stage["target"]][mask]

            print(
                f"Training {stage['stage']} on {dm.shape} data, y.sh = {y_stage.shape}"
            )
            if dm.empty:
                warnings.warn(
                    f"No samples to fit for stage: {str(stage_number)}",
                    f"{str(stage['stage'])}",
                    NoSamplesForStageWarning,
                )
                continue

            if self.classifier == "lr":
                clf = LogisticRegression(
                    random_state=42,
                    multi_class="multinomial",
                    class_weight="balanced",
                    solver="newton-cg",
                )
            elif self.classifier == "mlp":
                clf = MLPClassifier(alpha=1e-05, random_state=42)

            time_start = time()
            try:
                clf.fit(dm.to_numpy(), y_stage.to_numpy())
                stage["tree"].append(clf)
            except ValueError as err:
                print(f"ERROR training {stage['stage']}, {err}")
            finally:
                print(f"Stage {stage_number} trained, {time()-time_start} seconds")
        return self

    def get_rows_which_contain_labels(self, df, stage, selection, dm_cols):
        mask = df[stage["target"]].apply(
            lambda x: any(item for item in selection if item in x)
        )
        _df = df[mask]
        print("dm: with mask ", _df.shape)
        dm = _df[dm_cols]
        y_stage = df[stage["target"]][mask]

        return dm, y_stage

    def _check_fit(self):
        for stage in self.stages:
            if "tree" not in stage.keys():
                raise ClassifierNotFitError(
                    "Estimators not fitted, call `fit` before exploiting the model."
                )

    def _predict_stages(self, X):
        y_hats = None
        for stage_number, stage in enumerate(self.stages):
            if len(stage["tree"]) > 0:
                y_hat = stage["tree"][0].predict_proba(X)
                y_hat_p = stage["tree"][0].predict(X)

                # twists sometimes
                try:
                    pred_index = (
                        stage["tree"][0].classes_.tolist().index(stage["class"])
                    )
                    y_hat = y_hat[:, pred_index]
                    if y_hats is None:
                        y_hats = pd.DataFrame(y_hat, columns=[stage["target"]])
                    else:
                        y_hats[stage["target"]] = y_hat
                except Exception as ex:
                    print(f"ERROR predicting stage {stage['class']}")

        # map to column names where value > 0.5
        y_hat = y_hats.apply(
            lambda x: list(map(lambda x: x.split("_")[-1], x.index[x > 0.5].tolist())),
            1,
        )

        return y_hat

    def _predict_stages_topk(self, X, k):
        y_hats = None
        for stage_number, stage in enumerate(self.stages):
            if len(stage["tree"]) > 0:
                y_hat = stage["tree"][0].predict_proba(X)
                y_hat_p = stage["tree"][0].predict(X)

                # twists sometimes
                try:
                    pred_index = (
                        stage["tree"][0].classes_.tolist().index(stage["class"])
                    )
                    y_hat = y_hat[:, pred_index]
                    if y_hats is None:
                        y_hats = pd.DataFrame(y_hat, columns=[stage["target"]])
                    else:
                        y_hats[stage["target"]] = y_hat
                except Exception as ex:
                    print(f"ERROR predicting stage {stage['class']}")

        y_hat_topk = []
        for index, row in y_hats.iterrows():
            row_sorted = row.sort_values(ascending=False)
            head = row_sorted.head(k)
            topk = list(map(lambda x: x.split("_")[-1], head.index.tolist()))
            y_hat_topk.append(topk)

        return y_hat_topk

    def predict(self, X):
        """
        Predict class for X.
        """

        # Check that the trees have been fit
        self._check_fit()
        y_hat = self._predict_stages(X)
        return y_hat

    def predict_topk(self, X, k):
        """
        Predict class for X.
        """

        self._check_fit()
        y_hat = self._predict_stages_topk(X, k)
        return y_hat
