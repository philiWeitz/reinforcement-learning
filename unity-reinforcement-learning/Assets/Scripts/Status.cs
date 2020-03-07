using UnityEngine;

public class Status : MonoBehaviour
{
    public volatile bool isOnTrack = true;

    public volatile bool resetAgent = false;

    public volatile MoveModel networkMoveModel;

    public volatile bool shouldSendImage = true;

    public static volatile Status instance;

    public GameObject road;

    private Vector3[] vertices;


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

    public RoadPositionModel GetRandomRoadPosition()
    {
        if (vertices == null || vertices.Length <= 0)
        {
            Mesh mesh = road.GetComponent<MeshFilter>().mesh;
            vertices = mesh.vertices;
        }

        int idx = Random.Range(0, vertices.Length - 1);
        return new RoadPositionModel(vertices[idx], vertices[idx + 1]);
    }
}
