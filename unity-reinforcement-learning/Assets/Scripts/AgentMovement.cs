using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentMovement : MonoBehaviour
{
    public float speed = 6.0f;

    public float gravity = 20.0f;

    public volatile MoveModel moveModelFromPython = null;

    private CharacterController characterController;

    private int isNotGroundedCount;

    private Vector3 initialPosition;

    private Quaternion initialRotation;

    private Vector3 moveDirection = Vector3.zero;

    public static AgentMovement instance;

    private void Awake()
    {
        if (AgentMovement.instance == null)
        {
            AgentMovement.instance = this;
        }
        else if (AgentMovement.instance != this)
        {
            Destroy(this.gameObject);
        }
        DontDestroyOnLoad(this.gameObject);
    }

    void Start()
    {
        initialPosition = this.transform.position;
        initialRotation = this.transform.rotation;

        characterController = GetComponent<CharacterController>();
    }

    void Update()
    {
        if (moveModelFromPython == null)
        {
            MoveAgent(Input.GetAxisRaw("Horizontal"), Input.GetAxisRaw("Vertical"));
        }
        else
        {
            MakeMove(moveModelFromPython);
        }
    }

    void MoveAgent(float horizontal, float vertical)
    {
        if (!characterController.isGrounded)
        {
            isNotGroundedCount += 1;
        } else
        {
            isNotGroundedCount = 0;
        }

        if (isNotGroundedCount < 2)
        {
            if (characterController.isGrounded)
            {
                // We are grounded, so recalculate
                // move direction directly from axes
                moveDirection = new Vector3(0.0f, 0.0f, vertical);
                moveDirection *= speed;
            }

            Vector3 rotation = new Vector3(0, horizontal * 90 * Time.deltaTime, 0);
            this.transform.Rotate(rotation);

            // Apply gravity. Gravity is multiplied by deltaTime twice (once here, and once below
            // when the moveDirection is multiplied by deltaTime). This is because gravity should be applied
            // as an acceleration (ms^-2)
            moveDirection.y -= gravity * Time.deltaTime;
            moveDirection = this.transform.TransformDirection(moveDirection);

            characterController.Move(moveDirection * Time.deltaTime);
        }
        else
        {
            this.transform.rotation = initialRotation;
            this.transform.position = initialPosition;
            isNotGroundedCount = 0;
        }
    }

    public void MakeMove(MoveModel moveModel)
    {
        float horizontal = 0f;
        float vertical = 0f;

        if (moveModel.horizontal == "LEFT")
        {
            horizontal = -1;
        }
        if (moveModel.horizontal == "RIGHT")
        {
            horizontal = 1;
        }

        if (moveModel.vertical == "FORWARD")
        {
            vertical = 1;
        }
        if (moveModel.vertical == "BACKWARDS")
        {
            vertical = -1;
        }

        MoveAgent(horizontal, vertical);
    }
}
