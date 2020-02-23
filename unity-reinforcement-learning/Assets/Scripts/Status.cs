using UnityEngine;

public class Status : MonoBehaviour
{
    public volatile bool isSimulationRunning = true;

    public volatile bool isOnTrack = true;

    public volatile bool resetAgent = false;

    public volatile MoveModel networkMoveModel;

    public static volatile Status instance;


    private void Awake()
    {
        if (Status.instance == null)
        {
            Status.instance = this;
        }
        else if (Status.instance != this)
        {
            Destroy(this.gameObject);
        }
        DontDestroyOnLoad(this.gameObject);
    }
}
