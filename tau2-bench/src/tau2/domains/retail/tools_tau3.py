# Copyright Sierra
# Tau3-bench retail tools: extends RetailTools with the get_item_details
# read tool added in tau3-bench. Pulled verbatim from
# sierra-research/tau3-bench src/tau2/domains/retail/tools.py.
from tau2.domains.retail.data_model import Variant
from tau2.domains.retail.tools import RetailTools
from tau2.environment.toolkit import ToolType, is_tool


class RetailTau3Tools(RetailTools):
    def _get_item(self, item_id: str) -> Variant:
        """Get the item from the database.

        Args:
            item_id: The item id, such as '6086499569'. Be careful the item id is different from the product id.

        Returns:
            The item.

        Raises:
            ValueError: If the item is not found.
        """
        for _, product in self.db.products.items():
            if item_id in product.variants:
                return product.variants[item_id]

        raise ValueError("Item not found")

    @is_tool(ToolType.READ)
    def get_item_details(self, item_id: str) -> Variant:
        """Get the inventory details of an item.

        Args:
            item_id: The item id, such as '6086499569'. Be careful the item id is different from the product id.

        Returns:
            Variant: The item details.

        Raises:
            ValueError: If the item is not found.
        """
        item = self._get_item(item_id)
        return item
