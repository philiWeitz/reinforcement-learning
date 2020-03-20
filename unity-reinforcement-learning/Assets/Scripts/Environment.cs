﻿using UnityEngine;

public class Environment : MonoBehaviour
{
    public volatile bool isOnTrack = true;
    public volatile bool isFinishReached = false;
    public volatile bool isTerminalState = false;
    public volatile bool resetEnvironment = false;
    public volatile MoveModel networkMoveModel = new MoveModel();

    public static Environment instance;
    public volatile float gameSpeed = 1.0f;
    public volatile float timeOffTrackBeforeTerminalInSec = 2.0f;
    public volatile GameObject finishLine;
    public volatile GameObject trackMeshHolder;


    void FixedUpdate()
    {
        Time.timeScale = gameSpeed;
    }


    private void Awake()
    {
        if (Environment.instance == null)
        {
            Environment.instance = this;
        }
        else if (Environment.instance != this)
        {
            Destroy(this.gameObject);
        }
        DontDestroyOnLoad(this.gameObject);
    }
}