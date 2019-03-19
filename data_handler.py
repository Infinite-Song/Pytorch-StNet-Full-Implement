# coding=utf-8
import os
import io
import random
import json

labels = {'ApplyEyeMakeup': 0, 'ApplyLipstick': 1, 'Archery': 2, 'BabyCrawling': 3, 'BalanceBeam': 4, 'BandMarching': 5, 'BaseballPitch': 6, 'Basketball': 7, 'BasketballDunk': 8, 'BenchPress': 9, 'Biking': 10, 'Billiards': 11, 'BlowDryHair': 12, 'BlowingCandles': 13, 'BodyWeightSquats': 14, 'Bowling': 15, 'BoxingPunchingBag': 16, 'BoxingSpeedBag': 17, 'BreastStroke': 18, 'BrushingTeeth': 19, 'CleanAndJerk': 20, 'CliffDiving': 21, 'CricketBowling': 22, 'CricketShot': 23, 'CuttingInKitchen': 24, 'Diving': 25, 'Drumming': 26, 'Fencing': 27, 'FieldHockeyPenalty': 28, 'FloorGymnastics': 29, 'FrisbeeCatch': 30, 'FrontCrawl': 31, 'GolfSwing': 32, 'Haircut': 33, 'Hammering': 34, 'HammerThrow': 35, 'HandstandPushups': 36, 'HandstandWalking': 37, 'HeadMassage': 38, 'HighJump': 39, 'HorseRace': 40, 'HorseRiding': 41, 'HulaHoop': 42, 'IceDancing': 43, 'JavelinThrow': 44, 'JugglingBalls': 45, 'JumpingJack': 46, 'JumpRope': 47, 'Kayaking': 48, 'Knitting': 49, 'LongJump': 50, 'Lunges': 51, 'MilitaryParade': 52, 'Mixing': 53, 'MoppingFloor': 54, 'Nunchucks': 55, 'ParallelBars': 56, 'PizzaTossing': 57, 'PlayingCello': 58, 'PlayingDaf': 59, 'PlayingDhol': 60, 'PlayingFlute': 61, 'PlayingGuitar': 62, 'PlayingPiano': 63, 'PlayingSitar': 64, 'PlayingTabla': 65, 'PlayingViolin': 66, 'PoleVault': 67, 'PommelHorse': 68, 'PullUps': 69, 'Punch': 70, 'PushUps': 71, 'Rafting': 72, 'RockClimbingIndoor': 73, 'RopeClimbing': 74, 'Rowing': 75, 'SalsaSpin': 76, 'ShavingBeard': 77, 'Shotput': 78, 'SkateBoarding': 79, 'Skiing': 80, 'Skijet': 81, 'SkyDiving': 82, 'SoccerJuggling': 83, 'SoccerPenalty': 84, 'StillRings': 85, 'SumoWrestling': 86, 'Surfing': 87, 'Swing': 88, 'TableTennisShot': 89, 'TaiChi': 90, 'TennisSwing': 91, 'ThrowDiscus': 92, 'TrampolineJumping': 93, 'Typing': 94, 'UnevenBars': 95, 'VolleyballSpiking': 96, 'WalkingWithDog': 97, 'WallPushups': 98, 'WritingOnBoard': 99, 'YoYo': 100}

def txt2json(txtPath,jsonPath):
    Set = []
    if not os.path.exists(jsonPath):
        fobj = open(jsonPath, 'w')
        fobj.close()

    with open(txtPath, 'r') as f:
        dataList = f.readlines()
    for ele in dataList:
        Data = {}
        line = ele.split('\n')[0].split ('/')
        Data['video_name'] = line[1].split (' ')[0]
        Data['label'] = labels[line[0]]
        Set.append(Data)
        random.shuffle(Set)
    with open(jsonPath, 'w') as f:
        json.dump(Set, f)


if __name__ == '__main__':
    trainTxt = 'data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist01.txt'
    trainJson = 'data/train.json'
    testTxt = 'data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist01.txt'
    testJson = 'data/test.json'
    txt2json(trainTxt, trainJson)
    txt2json(testTxt,testJson)

    print('Done!')
