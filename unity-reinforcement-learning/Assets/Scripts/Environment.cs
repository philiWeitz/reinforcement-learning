using UnityEngine;

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

    public Light sun;
    public volatile bool enableSunTransition = true;


    void FixedUpdate()
    {
        Time.timeScale = gameSpeed;
    }


    private void Awake()
    {
        if (Environment.instance == null)
        {
            Environment.instance = this;
            InvokeRepeating("UpdateSunPosition", 1, 5);
        }
        else if (Environment.instance != this)
        {
            Destroy(this.gameObject);
        }
        DontDestroyOnLoad(this.gameObject);
    }

    private void UpdateSunPosition()
    {
        if (enableSunTransition && sun)
        {
            float angle = (sun.transform.rotation.eulerAngles.y + 1);
            if (angle > 300)
            {
                angle = 230;
            }
            sun.transform.rotation = Quaternion.Euler(
                sun.transform.rotation.eulerAngles.x, angle, sun.transform.rotation.eulerAngles.z);
        }
    }
}
