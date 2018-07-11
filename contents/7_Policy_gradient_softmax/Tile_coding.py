# -*- coding: utf-8 -*-
"""
# @Time    : 24/06/18 9:26 AM
# @Author  : ZHIMIN HOU
# @FileName: Tile_coding.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
"""
Tile Coding Software version 3.0beta
by Rich Sutton
based on a program created by Steph Schaeffer and others
External documentation and recommendations on the use of this code is available in the 
reinforcement learning textbook by Sutton and Barto, and on the web.
These need to be understood before this code is.

This software is for Python 3 or more.

This is an implementation of grid-style tile codings, based originally on
the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm), but by now highly changed. 
Here we provide a function, "tiles", that maps floating and integer
variables to a list of tiles, and a second function "tiles-wrap" that does the same while
wrapping some floats to provided widths (the lower wrap value is always 0).

The float variables will be gridded at unit intervals, so generalization
will be by approximately 1 in each direction, and any scaling will have 
to be done externally before calling tiles.

Num-tilings should be a power of 2, e.g., 16. To make the offsetting work properly, it should
also be greater than or equal to four times the number of floats.

The first argument is either an index hash table of a given size (created by (make-iht size)), 
an integer "size" (range of the indices from 0), or nil (for testing, indicating that the tile 
coordinates are to be returned without being converted to indices).
"""

import numpy as np
from math import floor, log
from itertools import zip_longest

basehash = hash


class IHT:

    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count(self):
        return len(self.dictionary)

    def fullp(self):
        return len(self.dictionary) >= self.size

    def getindex(self, obj, readonly=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif readonly:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount == 0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count


def hashcoords(coordinates, m, readonly=False):
    if type(m) == IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m) == int: return basehash(tuple(coordinates)) % m
    if m == None: return coordinates


def tiles(ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f * numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // numtilings)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))

    OneHotFeature = getOneHotFeature(ihtORsize.size, Tiles)
    return OneHotFeature


def tileswrap(ihtORsize, numtilings, floats, wrawidths, ints=[], readonly=False):

    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [floor(f * numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b % numtilings) // numtilings
            coords.append(c % width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles


def getOneHotFeature(maxsize, Tiles):

    feature = np.zeros(maxsize)
    for i in range(len(Tiles)):
        feature[Tiles[i]] = 1
    return feature


class Tilecoder:

    def __init__(self, env, numTilings, tilesPerTiling):
        # Set max value for normalization of inputs
        self.maxNormal = 1
        self.maxVal = env.observation_space.high
        # print(self.maxVal)
        self.minVal = env.observation_space.low
        # print(self.minVal)
        self.numTilings = numTilings
        self.tilesPerTiling = tilesPerTiling
        self.dim = len(self.maxVal)
        self.numTiles = (self.tilesPerTiling**self.dim) * self.numTilings
        self.actions = env.action_space.n
        self.n = self.numTiles * self.actions
        self.tileSize = np.divide(np.ones(self.dim)*self.maxNormal, self.tilesPerTiling-1)

    def getFeatures(self, variables):
        # Ensures range is always between 0 and self.maxValue
        values = np.zeros(self.dim)
        for i in range(len(self.maxVal)):
            values[i] = self.maxNormal * ((variables[i] - self.minVal[i])/(self.maxVal[i]-self.minVal[i]))
        tileIndices = np.zeros(self.numTilings)
        matrix = np.zeros([self.numTilings, self.dim])
        for i in range(self.numTilings):
            for i2 in range(self.dim):
                matrix[i, i2] = int(values[i2] / self.tileSize[i2] + i / self.numTilings)
        for i in range(1, self.dim):
            matrix[:, i] *= self.tilesPerTiling**i
        for i in range(self.numTilings):
            tileIndices[i] = (i * (self.tilesPerTiling**self.dim) + sum(matrix[i, :]))
        return tileIndices

    # def discretize(self, position, velocity, indices):
    #     position = (position - min(POSITION_RANGE)) / POSITION_RANGE_SIZE
    #     velocity = (velocity - min(VELOCITY_RANGE)) / VELOCITY_RANGE_SIZE
    #     for tiling in range(NUMBER_OF_TILINGS):
    #         offset = 0 if NUMBER_OF_TILINGS == 1 else tiling / \
    #                                                   float(NUMBER_OF_TILINGS)
    #
    #         position_index = int(position * (TILING_CARDINALITY - 1) + offset)
    #         position_index = min(position_index, TILING_CARDINALITY - 1)
    #
    #         velocity_index = int(velocity * (TILING_CARDINALITY - 1) + offset)
    #         velocity_index = min(velocity_index, TILING_CARDINALITY - 1)
    #
    #         indices[tiling] = position_index + velocity_index * \
    #                           TILING_CARDINALITY + TILING_AREA * tiling
    #
    #     return indices

    def oneHotVector(self, features, action):
        oneHot = np.zeros(self.n)
        for i in features:
            index = int(i + (self.numTiles*action))
            oneHot[index] = 1
        return oneHot

    def oneHotFeature(self, variables):
        features = self.getFeatures(variables)
        oneHot = np.zeros(self.numTiles)
        for i in features:
            index = int(i)
            oneHot[index] = 1
        return oneHot

    def getVal(self, theta, features, action):
        val = 0
        for i in features:
            index = int(i + (self.numTiles*action))
            val += theta[index]
        return val

    def getQ(self, features, theta):
        Q = np.zeros(self.actions)
        for i in range(self.actions):
            Q[i] = self.getVal(theta, features, i)
        return Q


"""
NUMBER_OF_TILINGS = 8
TILING_CARDINALITY = 9
POSITION_RANGE = (-1.2, 0.5)
VELOCITY_RANGE = (-0.07, 0.07)

TILING_AREA = TILING_CARDINALITY**2
TILE_COUNT = TILING_AREA * NUMBER_OF_TILINGS
POSITION_RANGE_SIZE = float(max(POSITION_RANGE) - min(POSITION_RANGE))
VELOCITY_RANGE_SIZE = float(max(VELOCITY_RANGE) - min(VELOCITY_RANGE))


def discretize(position, velocity, indices):
    position = (position - min(POSITION_RANGE)) / POSITION_RANGE_SIZE
    velocity = (velocity - min(VELOCITY_RANGE)) / VELOCITY_RANGE_SIZE
    for tiling in range(NUMBER_OF_TILINGS):

        offset = 0 if NUMBER_OF_TILINGS == 1 else tiling / \
            float(NUMBER_OF_TILINGS)

        position_index = int(position * (TILING_CARDINALITY - 1) + offset)
        position_index = min(position_index, TILING_CARDINALITY - 1)

        velocity_index = int(velocity * (TILING_CARDINALITY - 1) + offset)
        velocity_index = min(velocity_index, TILING_CARDINALITY - 1)

        indices[tiling] = position_index + velocity_index * \
            TILING_CARDINALITY + TILING_AREA * tiling
"""

