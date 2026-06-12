import json
import unittest

from backend.main import _build_task_from_intent
from backend.models.sarvam import _language_from_script
from backend.tasks.grocery import BlinkitPlannerTask, GroceryItem


class GroceryPlannerTests(unittest.TestCase):
    def test_intent_builds_task_and_normalizes_pack_quantity(self) -> None:
        task = _build_task_from_intent(
            {
                "task": "blinkit_planner",
                "grocery_mode": "meal_plan",
                "grocery_request": "Cook chicken biryani",
                "pantry_items": [],
                "grocery_items": [
                    {
                        "name": "Chicken",
                        "search_query": "fresh chicken",
                        "quantity": 500,
                        "required_amount": "500 g",
                        "reason": "Main protein",
                    }
                ],
            }
        )

        self.assertIsInstance(task, BlinkitPlannerTask)
        self.assertEqual(task.items[0].quantity, 1)
        self.assertEqual(task.items[0].required_amount, "500 g")

    def test_result_reports_added_and_unavailable_items(self) -> None:
        task = BlinkitPlannerTask(
            request="Make pasta",
            mode="missing_ingredients",
            pantry_items=["pasta"],
            items=[GroceryItem(name="Sauce", search_query="pasta sauce")],
        )
        payload = {
            "entries": [
                {
                    "requested_item": "Sauce",
                    "requested_amount": "1 bottle",
                    "requested_quantity": 1,
                    "product_name": "Tomato Pasta Sauce",
                    "pack_size": "500 g",
                    "price": "₹199",
                    "added_quantity": 1,
                    "status": "added",
                },
                {
                    "requested_item": "Cheese",
                    "requested_amount": "100 g",
                    "requested_quantity": 1,
                    "product_name": "",
                    "pack_size": "",
                    "price": "",
                    "added_quantity": 0,
                    "status": "unavailable",
                },
            ]
        }
        result = task.structured_result("GROCERY_PLANNER_DATA " + json.dumps(payload))

        self.assertEqual(result["added_count"], 1)
        self.assertEqual(result["unavailable_count"], 1)
        self.assertIn("Checkout was not opened", result["message"])

    def test_language_detection_uses_indian_script(self) -> None:
        self.assertEqual(_language_from_script("ಒಂದು ವಾರಕ್ಕೆ ದಿನಸಿ"), "kn-IN")
        self.assertEqual(_language_from_script("एक हफ्ते का सामान"), "hi-IN")
        self.assertIsNone(_language_from_script("make pasta"))


if __name__ == "__main__":
    unittest.main()
