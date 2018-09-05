# -*- coding: utf-8 -*-
"""
A classic (not left-leaning) Red-Black Tree implementation, supporting 
addition and deletion. The code is by Stanislav Kozlovski, modified
slightly by George Flanagin of University of Richmond.

Other than those changes, I left the code /in situ/.
"""

import sys
import typing
from typing import *

import enum
import traceback

__author__ = 'Stanislav Kozlovski'
__copyright__ = 'Copyright 2018, Stanislavkozlovski'
__version__ = '1.1'
__maintainer__ = 'George Flanagin'
__email__ = 'gflanagin@richmond.edu'


class NODE_COLOR(enum.Enum):
    """
    A red-black tree needs RED and BLACK, as well as the special
    colorless color that can be painted over, NIL.
    """
    BLACK = 'BLACK'
    RED = 'RED'
    NIL = 'NIL'


class Node:
    """ For the purpose of forward references. """
    pass


class Node:
    """
    A tree is filled with nodes. This is a good use of
    __slots__ because there will probably be a great many
    Nodes in any useful Tree.
    """

    __slots__ = ['value', 'color', 'parent', 'left', 'right']

    def __init__(self, value: Any,
                 color: NODE_COLOR,
                 parent: Node = None,
                 left: Node = None,
                 right: Node = None) -> None:

        self.value = value
        self.color = color
        self.parent = parent
        self.left = left
        self.right = right

    """ Two @classmethods to build new nodes of a given color. """
    @classmethod
    def BlackNode(Node_cls, value: Any) -> Node:
        return Node(value, NODE_COLOR.BLACK, None, NIL_LEAF, NIL_LEAF)

    @classmethod
    def RedNode(Node_cls, value: Any, parent: Node) -> Node:
        return Node(value, NODE_COLOR.RED, parent, NIL_LEAF, NIL_LEAF)

    """ A fistful of @properties to reduce the inline comparisons. """
    @property
    def is_black(self) -> bool:
        return self.color == NODE_COLOR.BLACK

    @property
    def is_not_black(self) -> bool:
        return self.color != NODE_COLOR.BLACK

    @property
    def is_red(self) -> bool:
        return self.color == NODE_COLOR.RED

    @property
    def is_not_red(self) -> bool:
        return self.color != NODE_COLOR.RED

    @property
    def is_colored(self) -> bool:
        return self.color != NODE_COLOR.NIL

    @property
    def is_uncolored(self) -> bool:
        return self.color == NODE_COLOR.NIL

    def __bool__(self) -> bool:
        """ implicit version of is_colored property """
        return self.color != NODE_COLOR.NIL

    def __repr__(self) -> str:
        """ 
        Spit out a few things about the node to help with
        debugging.
        """
        return '{color} {val} Node'.format(color=self.color, val=self.value)

    def __str__(self) -> str:
        """
        For debugging purposes in depth.
        """
        return "id:{} L:{} <- {}:{} -> R:{} P:{}".format(
            id(self), id(self.left), self.value, self.color, id(
                self.right), id(self.parent)
        )

    def iter_dump(self) -> object:
        """
        Just like __iter__, except that we dump out the __str__ values.
        """
        if self.left.color != NODE_COLOR.NIL:
            yield from self.left.iter_dump()

        yield str(self) + (" ROOT NODE " if self.parent is None else "")

        if self.right.color != NODE_COLOR.NIL:
            yield from self.right.iter_dump()

    def __iter__(self) -> object:
        """ 
        return all the left side Nodes, then one's self,
        then all the right side Nodes.
        """

        if self.left.color != NODE_COLOR.NIL:
            yield from self.left.__iter__()

        yield self.value

        if self.right.color != NODE_COLOR.NIL:
            yield from self.right.__iter__()

    def __eq__(self, other: Node):
        """
        Deciding if two nodes are the same based on their
        values and colors.
        """

        # Case 0: The argument is not a fellow Node.
        if not isinstance(other, Node):
            return NotImplemented

        # Case 1: Both nodes are NIL, so we don't need to check values.
        if self.color == NODE_COLOR.NIL and self.color == other.color:
            return True

        # Case 2: Do we have the same parent? If either we or they are
        # an orphan/root node, then we only need to check the other's
        # parents. Otherwise we need same parent and same value and color.
        if self.parent is None or other.parent is None:
            parents_are_same = self.parent is None and other.parent is None
        else:
            parents_are_same = (self.parent.value == other.parent.value
                                and self.parent.color == other.parent.color)

        return (self.value == other.value
                and self.color == other.color
                and parents_are_same)

    def has_children(self) -> bool:
        """ Returns a boolean indicating if the node has children """

        return bool(self.get_children_count())

    def get_children_count(self) -> int:
        """ Returns the number of NOT NIL children the node has """

        if self.color == NODE_COLOR.NIL:
            return 0
        return sum([int(self.left.color != NODE_COLOR.NIL), int(self.right.color != NODE_COLOR.NIL)])


NIL_LEAF = Node(value=None, color=NODE_COLOR.NIL, parent=None)


class RedBlackTree:
    """
    This is an implementation of the data structure described in the paper
    Leonidas J. Guibas and Robert Sedgewick (1978). "A Dichromatic 
    Framework for Balanced Trees". Proceedings of the 19th Annual 
    Symposium on Foundations of Computer Science. pp. 8–21. 
            doi:10.1109/SFCS.1978.3
    """

    def __init__(self) -> None:
        self.count = 0
        self.root = None
        self.value_type = type(None)

        # Used for deletion and uses the sibling's relationship with
        # his parent as a guide to the rotation
        self.ROTATIONS = {
            'L': self._right_rotation,
            'R': self._left_rotation
        }

    def __iter__(self) -> Node:
        """
        Work our way through the Nodes by invoking
        their __iter__ functions.
        """

        if not self.root:
            return []
        yield from self.root.__iter__()

    def iter_dump(self) -> str:
        if not self.root:
            return [""]
        yield from self.root.iter_dump()

    def __len__(self) -> int:
        return self.count

    def __stack_trace(self) -> str:
        """ Easy to read, tabular output. """

        exc_type, exc_value, exc_traceback = sys.exc_info()
        this_trace = traceback.extract_tb(exc_traceback)
        r = []
        for i, _ in enumerate(this_trace):
            r.append("""{}: {}, line {}, fcn {}\n    context=>> {}""".format(
                i, _[0], _[1], _[2], _[3]
            ))
        return "\n".join(r)

    def add(self, value: Any) -> None:
        """
        Add value to tree by creating a Node to contain it, and inserting
        the node at the correct location in the tree.
        """
        if not self.root:
            """ The root Node is always BLACK """
            self.root = Node.BlackNode(value)
            self.count += 1
            self.value_type = type(value)
            return

        if type(value) != self.value_type:
            raise Exception(
                "This tree contains {}. You cannot insert <{}> because it is a {}".format(
                    self.value_type, value, type(value)
                ))

        parent, node_dir = self._find_parent(value)

        if node_dir is None:  # value is in the tree
            return

        new_node = Node.RedNode(value, parent)
        if node_dir == 'L':
            parent.left = new_node
        else:
            parent.right = new_node

        self._try_rebalance(new_node)
        self.count += 1

    def remove(self, value: Any):
        """
        Try to get a node with 0 or 1 children.
        Either the node we're given has 0 or 1 children or we get its successor.
        """
        node_to_remove = self.find_node(value)
        if node_to_remove is None:  # node is not in the tree
            return

        if node_to_remove.get_children_count() == 2:
            # find the in-order successor and replace its value.
            # then, remove the successor
            successor = self._find_in_order_successor(node_to_remove)
            node_to_remove.value = successor.value  # switch the value
            node_to_remove = successor

        # Now it has 0 or 1 children! Remove it and decrement the census.
        self._remove(node_to_remove)
        self.count -= 1

    def contains(self, value: Any) -> bool:
        """ 
        Returns a boolean indicating if the given value is present in the tree 
        """

        return bool(self.find_node(value))

    def ceil(self, value: Any) -> Union[Node, None]:
        """
        Given a value, return the closest value such that value >= it,
        returning None when no such exists.
        """

        if self.root is None:
            return None
        last_found_val = None if self.root.value < value else self.root.value

        def find_ceil(node: Node) -> Node:
            """
            Interior recursive function to do the find/compare
            """
            nonlocal last_found_val
            if node == NIL_LEAF:
                return None

            if node.value == value:
                last_found_val = node.value
                return node.value

            elif node.value < value:  # go right
                return find_ceil(node.right)

            else:  # this node is bigger, save its value and go left
                last_found_val = node.value

                return find_ceil(node.left)

        find_ceil(self.root)
        return last_found_val

    def floor(self, value: Any) -> Union[Node, None]:
        """
        Given a value, return the closest value that is equal or less than it,
        returning None when no such exists
        """
        if self.root is None:
            return None
        last_found_val = None if self.root.value > value else self.root.value

        def find_floor(node: Node) -> Node:
            """
            Similar to the recursion in find_ceil()
            """
            nonlocal last_found_val
            if node == NIL_LEAF:
                return None

            if node.value == value:
                last_found_val = node.value
                return node.value

            elif node.value < value:
                # this node is smaller, save its value and go right, trying to find a cloer one
                last_found_val = node.value
                return find_floor(node.right)

            else:
                return find_floor(node.left)

        find_floor(self.root)
        return last_found_val

    def _remove(self, node: Node) -> None:
        """
        Receives a node with 0 or 1 children (typically some sort of successor)
        and removes it according to its color/children
        :param node: Node with 0 or 1 children
        """
        left_child = node.left
        right_child = node.right

        not_nil_child = left_child if left_child != NIL_LEAF else right_child
        if node == self.root:
            if not_nil_child != NIL_LEAF:
                # if we're removing the root and it has one valid child,
                # simply make that child the root
                self.root = not_nil_child
                self.root.parent = None
                self.root.color = NODE_COLOR.BLACK

            else:
                self.root = None

        elif node.is_red:
            if not node.has_children():
                # Red node with no children, the simplest remove
                self._remove_leaf(node)

            else:
                """
                Since the node is red he cannot have a child.
                If he had a child, it'd need to be black, but that would mean that
                the black height would be bigger on the one side and that would make our tree invalid
                """
                raise Exception("Red node with child: {}\n{}".format(
                    node, self.__stack_trace))

        else:  # node is black
            if right_child.has_children() or left_child.has_children():  # sanity check
                s = ('The red child of a black node with 0 or 1 children' +
                     ' cannot have children, otherwise the black height of' +
                     ' the tree becomes invalid!\n')
                raise(s + self.__stack_trace())

            if not_nil_child.color == NODE_COLOR.RED:
                """
                Swap the values with the red child and remove it  (basically un-link it)
                Since we're a node with one child only, we can be sure that there are no nodes below the red child.
                """
                node.value = not_nil_child.value
                node.left = not_nil_child.left
                node.right = not_nil_child.right

            else:  # NODE_COLOR.BLACK child
                self._remove_black_node(node)

    def _remove_leaf(self, leaf: Node) -> None:
        """ Simply removes a leaf node by making it's parent point to a NODE_COLOR.NIL LEAF"""
        if leaf.value >= leaf.parent.value:
            # in those weird cases where they're equal due to the successor swap
            leaf.parent.right = NIL_LEAF
        else:
            leaf.parent.left = NIL_LEAF

    def _remove_black_node(self, node: Node) -> None:
        """
        Loop through each case recursively until we reach a terminating case.
        What we're left with is a leaf node which is ready to be deleted without consequences
        """
        self.__case_1(node)
        self._remove_leaf(node)

    def __case_1(self, node: Node) -> None:
        """
        Case 1 is when there's a double black node on the root
        Because we're at the root, we can simply remove it
        and reduce the black height of the whole tree.

            __|10B|__                  __10B__
           /         \      ==>       /       \
          9B         20B            9B        20B
        """
        if self.root == node:
            node.color = NODE_COLOR.BLACK
            return

        self.__case_2(node)

    def __case_2(self, node: Node) -> None:
        """
        Case 2 applies when
            the parent is NODE_COLOR.BLACK
            the sibling is NODE_COLOR.RED
            the sibling's children are NODE_COLOR.BLACK or NODE_COLOR.NIL
        It takes the sibling and rotates it

                         40B                                              60B
                        /   \       --CASE 2 ROTATE-->                   /   \
                    |20B|   60R       LEFT ROTATE                      40R   80B
    DBL BLACK IS 20----^   /   \      SIBLING 60R                     /   \
                         50B    80B                                |20B|  50B
            (if the sibling's direction was left of it's parent, we would RIGHT ROTATE it)
        Now the original node's parent is NODE_COLOR.RED
        and we can apply case 4 or case 6
        """
        parent = node.parent
        sibling, direction = self._get_sibling(node)
        if silbing.is_red and parent.is_black and sibling.left.is_red and sibling.right.is_not_red:
            self.ROTATIONS[direction](
                node=None, parent=sibling, grandfather=parent)
            parent.color = NODE_COLOR.RED
            sibling.color = NODE_COLOR.BLACK
            return self.__case_1(node)
        self.__case_3(node)

    def __case_3(self, node: Node) -> None:
        """
        Case 3 deletion is when:
            the parent is NODE_COLOR.BLACK
            the sibling is NODE_COLOR.BLACK
            the sibling's children are NODE_COLOR.BLACK
        Then, we make the sibling red and
        pass the double black node upwards

                            Parent is black
               ___50B___    Sibling is black                       ___50B___
              /         \   Sibling's children are black          /         \
           30B          80B        CASE 3                       30B        |80B|  Continue with other cases
          /   \        /   \        ==>                        /  \        /   \
        20B   35R    70B   |90B|<---REMOVE                   20B  35R     70R   X
              /  \                                               /   \
            34B   37B                                          34B   37B
        """
        parent = node.parent
        sibling, _ = self._get_sibling(node)
        if sibling.is_black and parent.is_black and sibling.left.is_not_red and sibling.right.is_not_red:
            # color the sibling red and forward the double black node upwards
            # (call the cases again for the parent)
            sibling.color = NODE_COLOR.RED
            return self.__case_1(parent)  # start again

        self.__case_4(node)

    def __case_4(self, node: Node) -> None:
        """
        If the parent is red and the sibling is black with no red children,
        simply swap their colors
        DB-Double Black
                __10R__                   __10B__        The black height of the left subtree has been incremented
               /       \                 /       \       And the one below stays the same
             DB        15B      ===>    X        15R     No consequences, we're done!
                      /   \                     /   \
                    12B   17B                 12B   17B
        """
        parent = node.parent
        if parent.is_red:
            sibling, direction = self._get_sibling(node)
            if sibling.is_black and sibling.left.is_not_red and sibling.right.is_not_red:
                parent.color, sibling.color = sibling.color, parent.color  # switch colors
                return  # Terminating
        self.__case_5(node)

    def __case_5(self, node: Node) -> None:
        """
        Case 5 is a rotation that changes the circumstances so that we can do a case 6
        If the closer node is red and the outer NODE_COLOR.BLACK or NODE_COLOR.NIL, we do a left/right rotation, depending on the orientation
        This will showcase when the CLOSER NODE's direction is RIGHT

              ___50B___                                                    __50B__
             /         \                                                  /       \
           30B        |80B|  <-- Double black                           35B      |80B|        Case 6 is now
          /  \        /   \      Closer node is red (35R)              /   \      /           applicable here,
        20B  35R     70R   X     Outer is black (20B)               30R    37B  70R           so we redirect the node
            /   \                So we do a LEFT ROTATION          /   \                      to it :)
          34B  37B               on 35R (closer node)           20B   34B
        """
        sibling, direction = self._get_sibling(node)
        closer_node = sibling.right if direction == 'L' else sibling.left
        outer_node = sibling.left if direction == 'L' else sibling.right
        if closer_node.is_red and outer_node.is_not_red and sibling.is_black:
            if direction == 'L':
                self._left_rotation(
                    node=None, parent=closer_node, grandfather=sibling)
            else:
                self._right_rotation(
                    node=None, parent=closer_node, grandfather=sibling)
            closer_node.color = NODE_COLOR.BLACK
            sibling.color = NODE_COLOR.RED

        self.__case_6(node)

    def __case_6(self, node: Node) -> None:
        """
        Case 6 requires
            SIBLING to be NODE_COLOR.BLACK
            OUTER NODE to be NODE_COLOR.RED
        Then, does a right/left rotation on the sibling
        This will showcase when the SIBLING's direction is LEFT

                            Double Black
                    __50B__       |                               __35B__
                   /       \      |                              /       \
      SIBLING--> 35B      |80B| <-                             30R       50R
                /   \      /                                  /   \     /   \
             30R    37B  70R   Outer node is NODE_COLOR.RED            20B   34B 37B    80B
            /   \              Closer node doesn't                           /
         20B   34B                 matter                                   70R
                               Parent doesn't
                                   matter
                               So we do a right rotation on 35B!
        """
        sibling, direction = self._get_sibling(node)
        outer_node = sibling.left if direction == 'L' else sibling.right

        def __case_6_rotation(direction):
            parent_color = sibling.parent.color
            self.ROTATIONS[direction](
                node=None, parent=sibling, grandfather=sibling.parent)
            # new parent is sibling
            sibling.color = parent_color
            sibling.right.color = NODE_COLOR.BLACK
            sibling.left.color = NODE_COLOR.BLACK

        if sibling.is_black and outer_node.is_red:
            return __case_6_rotation(direction)  # terminating

        raise Exception('We should have ended here, something is wrong')

    def _try_rebalance(self, node: Node) -> None:
        """
        Given a red child node, determine if there is a need to rebalance (if the parent is red)
        If there is, rebalance it
        """
        parent = node.parent
        value = node.value

        if (parent is None or        # what the fuck? (should not happen)
            parent.parent is None or  # parent is the root
                (node.is_not_red or parent.is_not_red)):  # nothing to do.
            return

        """ 
        Take a look at the family tree to get info for the correct
        method of rebalancing. 
        """
        grandfather = parent.parent
        node_dir = 'L' if parent.value > value else 'R'
        parent_dir = 'L' if grandfather.value > parent.value else 'R'
        uncle = grandfather.right if parent_dir == 'L' else grandfather.left
        general_direction = node_dir + parent_dir

        if uncle == NIL_LEAF or uncle.is_black:
            # rotate
            if general_direction == 'LL':
                self._right_rotation(
                    node, parent, grandfather, to_recolor=True)

            elif general_direction == 'RR':
                self._left_rotation(node, parent, grandfather, to_recolor=True)

            elif general_direction == 'LR':
                self._right_rotation(
                    node=None, parent=node, grandfather=parent)
                # due to the prev rotation, our node is now the parent
                self._left_rotation(node=parent, parent=node,
                                    grandfather=grandfather, to_recolor=True)

            elif general_direction == 'RL':
                self._left_rotation(node=None, parent=node, grandfather=parent)
                # due to the prev rotation, our node is now the parent
                self._right_rotation(
                    node=parent, parent=node, grandfather=grandfather, to_recolor=True)

            else:
                raise Exception(
                    "{} is not a valid direction!".format(general_direction))

        else:  # uncle is NODE_COLOR.RED
            self._recolor(grandfather)

    def __update_parent(self, node: Node, parent_old_child: Node, new_parent: Node) -> None:
        """
        Our node 'switches' places with the old child
        Assigns a new parent to the node.
        If the new_parent is None, this means that our 
        node becomes the root of the tree
        """

        node.parent = new_parent
        if new_parent:
            # Determine the old child's position in order to put node there
            if new_parent.value > parent_old_child.value:
                new_parent.left = node
            else:
                new_parent.right = node
        else:
            self.root = node

    def _right_rotation(self, node: Node,
                        parent: Node,
                        grandfather: Node,
                        to_recolor: bool = False) -> None:

        grand_grandfather = grandfather.parent
        self.__update_parent(
            node=parent, parent_old_child=grandfather, new_parent=grand_grandfather)

        old_right = parent.right
        parent.right = grandfather
        grandfather.parent = parent

        grandfather.left = old_right  # save the old right values
        old_right.parent = grandfather

        if to_recolor:
            parent.color = NODE_COLOR.BLACK
            node.color = NODE_COLOR.RED
            grandfather.color = NODE_COLOR.RED

    def _left_rotation(self, node: Node,
                       parent: Node,
                       grandfather: Node,
                       to_recolor: bool = False) -> None:

        grand_grandfather = grandfather.parent
        self.__update_parent(
            node=parent, parent_old_child=grandfather, new_parent=grand_grandfather)

        old_left = parent.left
        parent.left = grandfather
        grandfather.parent = parent

        grandfather.right = old_left  # save the old left values
        old_left.parent = grandfather

        if to_recolor:
            parent.color = NODE_COLOR.BLACK
            node.color = NODE_COLOR.RED
            grandfather.color = NODE_COLOR.RED

    def _recolor(self, grandfather: Node) -> None:

        grandfather.right.color = NODE_COLOR.BLACK
        grandfather.left.color = NODE_COLOR.BLACK
        if grandfather != self.root:
            grandfather.color = NODE_COLOR.RED
        self._try_rebalance(grandfather)

    def _find_parent(self, value: Any) -> Node:
        """ Finds a place for the value in our binary tree"""

        def inner_find(parent):
            """
            Return the appropriate parent node for our new node 
            as well as the side it should be on
            """
            if value == parent.value:
                return None, None
            elif parent.value < value:
                if parent.right.is_uncolored:
                    return parent, 'R'
                return inner_find(parent.right)
            elif value < parent.value:
                if parent.left.is_uncolored:
                    return parent, 'L'
                return inner_find(parent.left)

        return inner_find(self.root)

    def find_node(self, value: Any) -> Node:
        def inner_find(root):
            if root is None or root == NIL_LEAF:
                return None
            if value > root.value:
                return inner_find(root.right)
            elif value < root.value:
                return inner_find(root.left)
            else:
                return root

        found_node = inner_find(self.root)
        return found_node

    def _find_in_order_successor(self, node: Node) -> Node:
        right_node = node.right
        left_node = right_node.left
        if left_node == NIL_LEAF:
            return right_node
        while left_node.left != NIL_LEAF:
            left_node = left_node.left
        return left_node

    def _get_sibling(self, node: Node) -> Tuple[Node, str]:
        """
        Returns the sibling of the node, as well as the side it is on
        e.g

            20 (A)
           /     \
        15(B)    25(C)

        _get_sibling(25(C)) => 15(B), 'R'
        """
        parent = node.parent
        if node.value >= parent.value:
            sibling = parent.left
            direction = 'L'
        else:
            sibling = parent.right
            direction = 'R'
        return sibling, direction


if __name__ == '__main__':
    import random
    import rb_tree

    t = rb_tree.RedBlackTree()
    for i in range(0, 30):
        v = random.choice(range(-100, 100))
        print("adding {}".format(v))
        t.add(v)

    print("\n".join([_ for _ in t.iter_dump()]))
