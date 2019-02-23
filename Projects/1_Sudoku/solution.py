# Stephen Blystone
# AIND Project 1 - Sudoku
#
# Implemented the naked twins strategy, and then generalized it to account for
# naked triplets, quadruplets, etc.
#
from copy import deepcopy
from utils import *


row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
unitlist = row_units + column_units + square_units

# Updated the unit list to add the new diagonal units
unitlist.extend([["A1", "B2", "C3", "D4", "E5", "F6", "G7", "H8", "I9"],
    ["A9", "B8", "C7", "D6", "E5", "F4", "G3", "H2", "I1"]])

# Must be called after all units (including diagonals) are added to the unitlist
units = extract_units(unitlist, boxes)
peers = extract_peers(units, boxes)


def naked_twins(values):
    """Eliminate values using the naked twins strategy.

    The naked twins strategy says that if you have two or more unallocated boxes
    in a unit and there are only two digits that can go in those two boxes, then
    those two digits can be eliminated from the possible assignments of all other
    boxes in the same unit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the naked twins eliminated from peers

    Notes
    -----
    Your solution can either process all pairs of naked twins from the input once,
    or it can continue processing pairs of naked twins until there are no such
    pairs remaining -- the project assistant test suite will accept either
    convention. However, it will not accept code that does not process all pairs
    of naked twins from the original input. (For example, if you start processing
    pairs of twins and eliminate another pair of twins before the second pair
    is processed then your code will fail the PA test suite.)

    The first convention is preferred for consistency with the other strategies,
    and because it is simpler (since the reduce_puzzle function already calls this
    strategy repeatedly).

    See Also
    --------
    Pseudocode for this algorithm on github:
    https://github.com/udacity/artificial-intelligence/blob/master/Projects/1_Sudoku/pseudocode.md
    """

    # function NakedTwins(values) returns a dict mapping from Sudoku box names to a list of feasible values
    #  inputs:
    #   values, a dict mapping from Sudoku box names to a list of feasible values
    #
    #  out <- copy(values) /* make a deep copy */
    #  for each boxA in values do
    #   for each boxB of PEERS(boxA) do
    #    if both values[boxA] and values[boxB] exactly match and have only two feasible digits do
    #     for each peer of INTERSECTION(PEERS(boxA), PEERS(boxB)) do
    #      for each digit of values[boxA] do
    #       remove digit d from out[peer]
    #  return out

    # Make a deep copy.
    # This treats the input as immutable to prevent unexpected results.
    out = deepcopy(values)

    # Go through each box in the board.
    for boxA, valA in values.items():
        # Don't run naked algorithm if there is a single value in boxA.
        if len(valA) == 1:
            continue

        # List to hold boxes with the same values that are peers with A, or A itself.
        # Initialize to A. Cannot call list(boxA) because it breaks the string into
        # separate values.  Example: 'A1' becomes ['A', '1'] when calling list('A1').
        boxListSameVal = [boxA]

        # For each peer, if the values match, then add box to boxListSameVal.
        for boxB in peers[boxA]:
            if (valA == values[boxB]) and boxB not in boxListSameVal:
                boxListSameVal.append(boxB)

        # If number of possible values equals the number of boxes,
        # then this is candidate for naked twins/triplets/quadruplets/etc.
        if len(valA) == len(boxListSameVal):
            # Initialize our set of commonPeers to be the peers of boxA.
            commonPeers = set(peers[boxA])
            for b in boxListSameVal:
                # Find the intersection of peers for all boxes.
                commonPeers.intersection_update(commonPeers, peers[b])

            # Update those commonPeers by removing the values in the
            # naked twins/triplets/quadruplets/etc.
            for peer in list(commonPeers):
                for digit in valA:
                    out[peer] = out[peer].replace(digit, '')

    return out


def eliminate(values):
    """Apply the eliminate strategy to a Sudoku puzzle

    The eliminate strategy says that if a box has a value assigned, then none
    of the peers of that box can have the same value.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the assigned values eliminated from peers
    """

    # Go through each box.
    for box, val in values.items():
        # If the box has a single value.
        if len(val) == 1:
            # For each peer of box, remove val from peer's possible values.
            for peer in peers[box]:
                values[peer] = values[peer].replace(val, '')

    return values



def only_choice(values):
    """Apply the only choice strategy to a Sudoku puzzle

    The only choice strategy says that if only one box in a unit allows a certain
    digit, then that box must be assigned that digit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with all single-valued boxes assigned

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    """
    # For each unit.
    for unit in unitlist:
        # Go through each number.
        for num in '123456789':
            # Count the number of occurrences of a number occurring in a unit.
            numOccurrences = [box for box in unit if num in values[box]]
            # If it is the only choice, then mark the value as that number.
            if len(numOccurrences) == 1:
                values[numOccurrences[0]] = num
    return values


def reduce_puzzle(values):
    """Reduce a Sudoku puzzle by repeatedly applying all constraint strategies

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary after continued application of the constraint strategies
        no longer produces any changes, or False if the puzzle is unsolvable
    """
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])

        # Use the Eliminate Strategy
        values = eliminate(values)

        # Use the Only Choice Strategy
        values = only_choice(values)

        # Use the Naked Twins Strategy
        values = naked_twins(values)

        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    """Apply depth first search to solve Sudoku puzzles in order to solve puzzles
    that cannot be solved by repeated reduction alone.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary with all boxes assigned or False

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    and extending it to call the naked twins strategy.
    """
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)

    if values is False:
        # Box found with 0 possible values.
        return False

    # Check if solved.
    if all(len(values[box]) == 1 for box in boxes):
        return values

    # Choose one of the unfilled squares with the fewest possibilities.
    minBox = min((len(values[box]), box) for box in values if len(values[box]) > 1)[1]

    # Now use recursion to solve each one of the resulting sudokus,
    # and if one returns a value (not False), return that answer!
    for val in values[minBox]:
        # Make a deep copy.
        tmpValues = deepcopy(values)
        # Change the value.
        tmpValues[minBox] = val
        # Search to see if a solution is found.
        retVal = search(tmpValues)
        # If so, then return the solution.
        if retVal:
            return retVal


def solve(grid):
    """Find the solution to a Sudoku puzzle using search and constraint propagation

    Parameters
    ----------
    grid(string)
        a string representing a sudoku grid.

        Ex. '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'

    Returns
    -------
    dict or False
        The dictionary representation of the final sudoku grid or False if no solution exists.
    """
    values = grid2values(grid)
    values = search(values)
    return values


if __name__ == "__main__":
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(grid2values(diag_sudoku_grid))
    result = solve(diag_sudoku_grid)
    display(result)

    try:
        import PySudoku
        PySudoku.play(grid2values(diag_sudoku_grid), result, history)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
