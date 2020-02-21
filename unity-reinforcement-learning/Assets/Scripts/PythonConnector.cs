using System;
using System.Net;
using System.Net.Sockets;
using UnityEngine;

public class PythonConnector : MonoBehaviour
{
    public int port = 11000;

    public Camera agentCamera;

    private byte[] _recieveBuffer = new byte[8142];

    private Socket clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);


    void Start()
    {
        InvokeRepeating("SendData", 1, 1);
    }

    void Update()
    {
        // always try to reconnect
        if (!clientSocket.IsBound)
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
        catch (SocketException ex) {}
    }

    private void ReceiveCallback(IAsyncResult AR)
    {
        int recieved = clientSocket.EndReceive(AR);

        if (recieved > 0) {
            byte[] jsonData = new byte[recieved];
            Buffer.BlockCopy(_recieveBuffer, 0, jsonData, 0, recieved);

            MoveModel moveModel = MoveModel.FromJsonBytes(jsonData);
            AgentMovement.instance.MakeMove(moveModel);
        }

        clientSocket.BeginReceive(_recieveBuffer, 0, _recieveBuffer.Length, SocketFlags.None, new AsyncCallback(ReceiveCallback), null);
    }

    private void SendData()
    {
        if (clientSocket.Connected)
        {
            Texture2D agentTexture = GetAgentCameraTexture();

            if (agentTexture)
            {
                Color[] colors = agentTexture.GetPixels();
                FrameModel frame = new FrameModel("Hello Frame", colors);

                byte[] data = frame.ToJsonBytes();
                Destroy(agentTexture);

                SocketAsyncEventArgs socketAsyncData = new SocketAsyncEventArgs();
                socketAsyncData.SetBuffer(data, 0, data.Length);
                clientSocket.SendAsync(socketAsyncData);
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
