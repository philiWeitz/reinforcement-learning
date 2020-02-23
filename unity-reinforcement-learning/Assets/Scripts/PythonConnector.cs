using System;
using System.Net;
using System.Net.Sockets;
using UnityEngine;
using System.Text;

public class PythonConnector : MonoBehaviour
{
    public int port = 11000;

    public Camera agentCamera;

    public float repeatEverySec = 0.5f;

    private byte[] _recieveBuffer = new byte[8142];

    private Socket clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);


    void Start()
    {
        InvokeRepeating("SendData", 1, repeatEverySec);
    }

    void Update()
    {
        // always try to reconnect
        if (!clientSocket.Connected)
        {
            SetupConnection();
        }
    }

    void OnDestroy()
    {
        if (clientSocket.Connected)
        {
            clientSocket.Disconnect(false);
        }
    }

    private void SetupConnection()
    {
        try
        {
            clientSocket.Connect(new IPEndPoint(IPAddress.Loopback, port));
            clientSocket.BeginReceive(_recieveBuffer, 0, _recieveBuffer.Length, SocketFlags.None, new AsyncCallback(ReceiveCallback), null);
        }
        catch (SocketException) {}
    }

    private void ReceiveCallback(IAsyncResult AR)
    {
        int recieved = clientSocket.EndReceive(AR);

        if (recieved > 0) {
            byte[] data = new byte[recieved];
            Buffer.BlockCopy(_recieveBuffer, 0, data, 0, recieved);

            string dataString = Encoding.UTF8.GetString(data);
            if (dataString == "RESET")
            {
                Status.instance.resetAgent = true;
            }
            else
            {
                // set the move model received by the socket
                MoveModel moveModel = MoveModel.FromJsonBytes(data);
                Status.instance.networkMoveModel = moveModel;
            }
        }
        // start receiving data again
        clientSocket.BeginReceive(_recieveBuffer, 0, _recieveBuffer.Length, SocketFlags.None, new AsyncCallback(ReceiveCallback), null);
    }

    private void SendData()
    {
        // only send picture when the simmulation is running
        if (clientSocket.Connected && Status.instance.isSimulationRunning)
        {
            Texture2D agentTexture = GetAgentCameraTexture();

            if (agentTexture)
            {
                Color[] colors = agentTexture.GetPixels();
                FrameModel frame = new FrameModel(Status.instance.isOnTrack, colors);

                byte[] data = frame.ToJsonBytes();
                Destroy(agentTexture);

                SocketAsyncEventArgs socketAsyncData = new SocketAsyncEventArgs();
                socketAsyncData.SetBuffer(data, 0, data.Length);
                clientSocket.SendAsync(socketAsyncData);

                // Stop simulation when the agent is off track
                if (!Status.instance.isOnTrack)
                {
                    Status.instance.isSimulationRunning = false;
                }
            }
        }
    }

    private Texture2D GetAgentCameraTexture()
    {
        Rect rect = new Rect(0, 0, 120, 50);
        RenderTexture renderTexture = new RenderTexture(120, 50, 24);
        Texture2D screenShot = new Texture2D(120, 50, TextureFormat.RGB24, false);

        agentCamera.targetTexture = renderTexture;
        agentCamera.Render();

        RenderTexture.active = renderTexture;
        screenShot.ReadPixels(rect, 0, 0);

        agentCamera.targetTexture = null;
        RenderTexture.active = null;

        Destroy(renderTexture);
        renderTexture = null;

        return screenShot;
    }
}
