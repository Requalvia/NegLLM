# -*- coding: utf-8 -*-
import numpy as np
import bisect

"""
Utility functions for antique negotiation with weighted preferences.
"""



y = 2.0    


def reward_normalize(reward):

    BIN_EDGES = [np.float64(-400.0), np.float64(-395.0), np.float64(-390.0), np.float64(-385.0), np.float64(-380.0), np.float64(-375.0), np.float64(-370.0), np.float64(-365.0), np.float64(-360.0), np.float64(-355.0), np.float64(-350.0), np.float64(-345.0), np.float64(-340.0), np.float64(-335.0), np.float64(-330.0), np.float64(-325.0), np.float64(-320.0), np.float64(-315.0), np.float64(-310.0), np.float64(-305.0), np.float64(-300.0), np.float64(-295.0), np.float64(-290.0), np.float64(-285.0), np.float64(-280.0), np.float64(-275.0), np.float64(-270.0), np.float64(-265.0), np.float64(-260.0), np.float64(-255.0), np.float64(-250.0), np.float64(-245.0), np.float64(-240.0), np.float64(-235.0), np.float64(-230.0), np.float64(-225.0), np.float64(-220.0), np.float64(-215.0), np.float64(-210.0), np.float64(-205.0), np.float64(-200.0), np.float64(-195.0), np.float64(-190.0), np.float64(-185.0), np.float64(-180.0), np.float64(-175.0), np.float64(-170.0), np.float64(-165.0), np.float64(-160.0), np.float64(-155.0), np.float64(-150.0), np.float64(-145.0), np.float64(-140.0), np.float64(-135.0), np.float64(-130.0), np.float64(-125.0), np.float64(-120.0), np.float64(-115.0), np.float64(-110.0), np.float64(-105.0), np.float64(-100.0), np.float64(-95.0), np.float64(-90.0), np.float64(-85.0), np.float64(-80.0), np.float64(-75.0), np.float64(-70.0), np.float64(-65.0), np.float64(-60.0), np.float64(-55.0), np.float64(-50.0), np.float64(-45.0), np.float64(-40.0), np.float64(-35.0), np.float64(-30.0), np.float64(-25.0), np.float64(-20.0), np.float64(-15.0), np.float64(-10.0), np.float64(-5.0), np.float64(0.0), np.float64(5.0), np.float64(10.0), np.float64(15.0), np.float64(20.0), np.float64(25.0), np.float64(30.0), np.float64(35.0), np.float64(40.0), np.float64(45.0), np.float64(50.0), np.float64(55.0), np.float64(60.0), np.float64(65.0), np.float64(70.0), np.float64(75.0), np.float64(80.0), np.float64(85.0), np.float64(90.0), np.float64(95.0), np.float64(100.0), np.float64(105.0), np.float64(110.0), np.float64(115.0), np.float64(120.0), np.float64(125.0), np.float64(130.0), np.float64(135.0), np.float64(140.0), np.float64(145.0), np.float64(150.0), np.float64(155.0), np.float64(160.0), np.float64(165.0), np.float64(170.0), np.float64(175.0), np.float64(180.0), np.float64(185.0), np.float64(190.0), np.float64(195.0), np.float64(200.0), np.float64(205.0), np.float64(210.0), np.float64(215.0), np.float64(220.0), np.float64(225.0), np.float64(230.0), np.float64(235.0), np.float64(240.0), np.float64(245.0), np.float64(250.0), np.float64(255.0), np.float64(260.0), np.float64(265.0), np.float64(270.0), np.float64(275.0), np.float64(280.0), np.float64(285.0), np.float64(290.0), np.float64(295.0), np.float64(300.0), np.float64(305.0), np.float64(310.0), np.float64(315.0), np.float64(320.0), np.float64(325.0), np.float64(330.0), np.float64(335.0), np.float64(340.0), np.float64(345.0), np.float64(350.0), np.float64(355.0), np.float64(360.0), np.float64(365.0), np.float64(370.0), np.float64(375.0), np.float64(380.0), np.float64(385.0), np.float64(390.0), np.float64(395.0), np.float64(400.0), np.float64(405.0), np.float64(410.0), np.float64(415.0), np.float64(420.0), np.float64(425.0), np.float64(430.0), np.float64(435.0), np.float64(440.0), np.float64(445.0), np.float64(450.0), np.float64(455.0), np.float64(460.0), np.float64(465.0), np.float64(470.0), np.float64(475.0), np.float64(480.0), np.float64(485.0), np.float64(490.0), np.float64(495.0), np.float64(500.0), np.float64(505.0), np.float64(510.0), np.float64(515.0), np.float64(520.0), np.float64(525.0), np.float64(530.0), np.float64(535.0), np.float64(540.0), np.float64(545.0), np.float64(550.0), np.float64(555.0), np.float64(560.0), np.float64(565.0), np.float64(570.0), np.float64(575.0), np.float64(580.0), np.float64(585.0), np.float64(590.0), np.float64(595.0), np.float64(600.0)]
    CUM_PROB = [0.000152, 0.000152, 0.000457, 0.000457, 0.000914, 0.000914, 0.001524, 0.001524, 0.002286, 0.002286, 0.003201, 0.003201, 0.004268, 0.004268, 0.005487, 0.005487, 0.006859, 0.006859, 0.008383, 0.008383, 0.010059, 0.010212, 0.012041, 0.012346, 0.014327, 0.014784, 0.016918, 0.017528, 0.019814, 0.020576, 0.023015, 0.023929, 0.02652, 0.027587, 0.030331, 0.03155, 0.034446, 0.035818, 0.038866, 0.04039, 0.043591, 0.045267, 0.048621, 0.05045, 0.053955, 0.055937, 0.059595, 0.061728, 0.065539, 0.067825, 0.071788, 0.074226, 0.078342, 0.080933, 0.0852, 0.087944, 0.092364, 0.09526, 0.099832, 0.102881, 0.107606, 0.110959, 0.115836, 0.119494, 0.124524, 0.128487, 0.133669, 0.137936, 0.143271, 0.147843, 0.15333, 0.158208, 0.163847, 0.169029, 0.174821, 0.180308, 0.186252, 0.192044, 0.198141, 0.204237, 0.210486, 0.216888, 0.223442, 0.230148, 0.237007, 0.244018, 0.251181, 0.258497, 0.265966, 0.273586, 0.28136, 0.289133, 0.297058, 0.304984, 0.313062, 0.32114, 0.329371, 0.337601, 0.345984, 0.354367, 0.362902, 0.371437, 0.380125, 0.388813, 0.397653, 0.406493, 0.415485, 0.424478, 0.433623, 0.442768, 0.452065, 0.46121, 0.470508, 0.479652, 0.48895, 0.498095, 0.507392, 0.516537, 0.525834, 0.534979, 0.544277, 0.553422, 0.562719, 0.571864, 0.581161, 0.590306, 0.599604, 0.608749, 0.618046, 0.627191, 0.636488, 0.645633, 0.654778, 0.663771, 0.672763, 0.681603, 0.690444, 0.699131, 0.707819, 0.716354, 0.724889, 0.733272, 0.741655, 0.749886, 0.758116, 0.766194, 0.774272, 0.782198, 0.790123, 0.797897, 0.80567, 0.813291, 0.820759, 0.828075, 0.835239, 0.84225, 0.849108, 0.855815, 0.862369, 0.86877, 0.875019, 0.881116, 0.88706, 0.892852, 0.898491, 0.903978, 0.909313, 0.914495, 0.919524, 0.924402, 0.929127, 0.933699, 0.938119, 0.942387, 0.946502, 0.950465, 0.954275, 0.957933, 0.961439, 0.964792, 0.967993, 0.971041, 0.973937, 0.97668, 0.979271, 0.98171, 0.983996, 0.98613, 0.988112, 0.989941, 0.991617, 0.993141, 0.994513, 0.995732, 0.996799, 0.997714, 0.998476, 0.999086, 0.999543, 1.0, 1.0]


    if isinstance(reward, dict):
        reward = list(reward.values())[0]
    if not isinstance(reward, (int, float)):
        reward = reward[0]
    r = float(reward)

    # situations beyond the known bins
    if r <= BIN_EDGES[0]:
        return 0.0
    if r >= BIN_EDGES[-1]:
        return 1.0


    i = bisect.bisect_right(BIN_EDGES, r) - 1


    lb = BIN_EDGES[i]
    rb = BIN_EDGES[i+1]


    frac = (r - lb) / (rb - lb)


    p = CUM_PROB[i] + frac * (CUM_PROB[i+1] - CUM_PROB[i])

    return float(round(np.clip(p, 0.0, 1.0), 3))



def offer_normalize(offer, step=5):
    """
    Convert any offer into a discretized form consistent with pareto.json.
    This step is required before the offer can be used for querying in the JSON!
    - Values are rounded to the nearest multiple of the step.
    """

    return {item: int(round(price / step) * step) for item, price in offer.items()}


def uf_a(offer: dict):
    # print(offer)
    """
    Seller's perspective reward:
    - Weighted profit:
        ĻANTIQUE_1Ļ profit * (1 + x)
        ĻANTIQUE_2Ļ profit * (1 - x)
    - If profit < 0, apply ×y penalty.
    """
    try:
        reward = 0
        explanation_parts = []

        # ĻANTIQUE_1Ļ
        ĻANTIQUE_1Ļ_offer = offer["ĻANTIQUE_1Ļ"]
        cost_price_ĻANTIQUE_1Ļ = ĻANT_1_INĻ
        profit_ĻANTIQUE_1Ļ = ĻANTIQUE_1Ļ_offer - cost_price_ĻANTIQUE_1Ļ
        penalty_factor = y if profit_ĻANTIQUE_1Ļ < 0 else 1
        weighted_profit_ĻANTIQUE_1Ļ = profit_ĻANTIQUE_1Ļ * penalty_factor
        reward += weighted_profit_ĻANTIQUE_1Ļ
        explanation_parts.append(
            f"ĻANTIQUE_1Ļ: Profit {profit_ĻANTIQUE_1Ļ} "
            f"{' * y=' + str(y) + ' (loss penalty)' if penalty_factor > 1 else ''}"
            f" = {weighted_profit_ĻANTIQUE_1Ļ:.1f} (Bought for {cost_price_ĻANTIQUE_1Ļ}, sold for {ĻANTIQUE_1Ļ_offer})"
        )

        # ĻANTIQUE_2Ļ the antique 2
        ĻANTIQUE_2Ļ_offer = offer["ĻANTIQUE_2Ļ"]
        cost_price_ĻANTIQUE_2Ļ = ĻANT_2_INĻ
        profit_ĻANTIQUE_2Ļ = ĻANTIQUE_2Ļ_offer - cost_price_ĻANTIQUE_2Ļ
        penalty_factor = y if profit_ĻANTIQUE_2Ļ < 0 else 1
        weighted_profit_ĻANTIQUE_2Ļ = profit_ĻANTIQUE_2Ļ * penalty_factor
        reward += weighted_profit_ĻANTIQUE_2Ļ
        explanation_parts.append(
            f"ĻANTIQUE_2Ļ: Profit {profit_ĻANTIQUE_2Ļ} "
            f"{' * y=' + str(y) + ' (loss penalty)' if penalty_factor > 1 else ''}"
            f" = {weighted_profit_ĻANTIQUE_2Ļ:.1f} (Bought for {cost_price_ĻANTIQUE_2Ļ}, sold for {ĻANTIQUE_2Ļ_offer})"
        )

        explanation = " ; ".join(explanation_parts)
        return reward, explanation
    except Exception as e:
        print(e)
        print(offer)
        return 0,"Illegal offer"


def uf_b(offer: dict):
    # print(offer)
    """
    Buyer's perspective reward:
    - Weighted gain:
        ĻANTIQUE_1Ļ gain * (1 - x)
        ĻANTIQUE_2Ļ gain * (1 + x)
    - If gain < 0, apply ×y penalty.
    """
    try:
        reward = 0
        explanation_parts = []

        # ĻANTIQUE_1Ļ
        ĻANTIQUE_1Ļ_offer = offer["ĻANTIQUE_1Ļ"]
        market_price_ĻANTIQUE_1Ļ = ĻANT_1_OUTĻ
        gain_ĻANTIQUE_1Ļ = market_price_ĻANTIQUE_1Ļ - ĻANTIQUE_1Ļ_offer
        penalty_factor = y if gain_ĻANTIQUE_1Ļ < 0 else 1
        weighted_gain_ĻANTIQUE_1Ļ = gain_ĻANTIQUE_1Ļ* penalty_factor
        reward += weighted_gain_ĻANTIQUE_1Ļ
        explanation_parts.append(
            f"ĻANTIQUE_1Ļ: Gain {gain_ĻANTIQUE_1Ļ} "
            f"{' * y=' + str(y) + ' (loss penalty)' if penalty_factor > 1 else ''}"
            f" = {weighted_gain_ĻANTIQUE_1Ļ:.1f} (Bought for {ĻANTIQUE_1Ļ_offer}, market {market_price_ĻANTIQUE_1Ļ})"
        )

        # ĻANTIQUE_2Ļ
        ĻANTIQUE_2Ļ_offer = offer["ĻANTIQUE_2Ļ"]
        market_price_ĻANTIQUE_2Ļ = ĻANT_2_OUTĻ
        gain_ĻANTIQUE_2Ļ = market_price_ĻANTIQUE_2Ļ - ĻANTIQUE_2Ļ_offer
        penalty_factor = y if gain_ĻANTIQUE_2Ļ < 0 else 1
        weighted_gain_ĻANTIQUE_2Ļ = gain_ĻANTIQUE_2Ļ * penalty_factor
        reward += weighted_gain_ĻANTIQUE_2Ļ
        explanation_parts.append(
            f"ĻANTIQUE_2Ļ: Gain {gain_ĻANTIQUE_2Ļ} "
            f"{' * y=' + str(y) + ' (loss penalty)' if penalty_factor > 1 else ''}"
            f" = {weighted_gain_ĻANTIQUE_2Ļ:.1f} (Bought for {ĻANTIQUE_2Ļ_offer}, market {market_price_ĻANTIQUE_2Ļ})"
        )

        explanation = " ; ".join(explanation_parts)
        return reward, explanation
    except Exception as e:
        print(e)
        print(offer)
        return 0,"Illegal offer"
