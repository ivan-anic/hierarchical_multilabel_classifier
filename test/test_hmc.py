"""
Tests for the hmc module.
"""

import unittest

import pandas as pd
from sklearn.model_selection import train_test_split

import hmc
import hmcdatasets
import metrics


class TestClassHierarchy(unittest.TestCase):
    def test_count_classes(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        ch.print_()
        self.assertEqual(len(ch.classes_()), 8)

    def test_count_classes(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        ch.print_()
        self.assertEqual(len(ch.classes_()), 8)

    def test_count_nodes(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        self.assertEqual(len(ch.nodes_()), 7)

    def test_get_parent(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        self.assertEqual(ch._get_parent("black"), "dark")

    def test_get_children(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        self.assertEqual(ch._get_children("dark"), ["black", "gray"])

    def test_get_ancestors(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        self.assertEqual(ch._get_ancestors("ash"), ["gray", "dark"])
        self.assertEqual(len(ch._get_ancestors("colors")), 0)

    def test_get_descendants(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        self.assertEqual(ch._get_descendants("dark"), ["black", "gray", "ash", "slate"])
        self.assertEqual(len(ch._get_descendants("slate")), 0)

    def test_add_node(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        old_number = len(ch.nodes_())
        ch.add_node("additional node", ch.root)
        new_number = len(ch.nodes_())
        # Adding a node should increase node count by 1
        self.assertEqual(old_number + 1, new_number)

    def test_add_redundant_node(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        ch.add_node("redundant_node", ch.root)
        old_number = len(ch.nodes_())
        ch.add_node("redundant_node", ch.root)
        new_number = len(ch.nodes_())
        # Adding a redundant node should not increase node count
        self.assertEqual(old_number, new_number)

    def test_add_root_node(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        # Adding the root as a child should throw an exception
        self.assertRaises(ValueError, ch.add_node, "colors", "light")

    def test_add_dag_node(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        # Adding a child with a new parent should throw an exception
        self.assertRaises(ValueError, ch.add_node, "slate", "light")


class TestDecisionTreeHierarchicalClassifier(unittest.TestCase):
    def test_fit(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        ch.print_()
        X, y = hmcdatasets.load_shades_data()
        dt = hmc.DecisionTreeHierarchicalClassifier(ch)
        dt = dt.fit(X, y)
        trees_fit = True
        for stage in dt.stages:
            if "tree" not in stage.keys():
                trees_fit = False
        # After the fit each stage should have a tree
        self.assertEqual(trees_fit, True)

    def test_predict(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        X, y = hmcdatasets.load_shades_data()
        dt = hmc.DecisionTreeHierarchicalClassifier(ch)
        dt = dt.fit(X, y)
        predictions = dt.predict(X)
        # One prediction for each observation
        self.assertEqual(len(predictions), len(X))

    def test_score(self):
        ch = hmcdatasets.load_shades_class_hierachy()
        X, y = hmcdatasets.load_shades_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.50, random_state=0
        )
        dt = hmc.DecisionTreeHierarchicalClassifier(ch)
        dt = dt.fit(X_train, y_train)

        y_pred = dt.predict(X_test)

        metrics.classification_report(ch, y_test, pd.DataFrame(y_pred))


if __name__ == "__main__":
    unittest.main()
