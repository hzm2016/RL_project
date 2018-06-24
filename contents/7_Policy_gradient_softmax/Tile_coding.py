# -*- coding: utf-8 -*-
"""
# @Time    : 24/06/18 9:26 AM
# @Author  : ZHIMIN HOU
# @FileName: Tile_coding.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
import numpy as np


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