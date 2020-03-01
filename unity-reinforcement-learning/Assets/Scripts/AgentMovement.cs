using UnityEngine;

public class AgentMovement : MonoBehaviour
{
    public float speed = 12.0f;

    public float rotationSpeed = 180f;

    public float gravity = 20.0f;

    private Vector3 initialPosition;

    private Quaternion initialRotation;


    void Start()
    {
        initialPosition = this.transform.position;
        initialRotation = this.transform.rotation;
    }

    void Update()
    {
        if (Status.instance.resetAgent || Input.GetKeyUp("space"))
        {
            ResetAgent();
            return;
        }

        // only move the agent when the simmulation is running
        if (Status.instance.isOnTrack)
        {
            if (HasUserInput())
            {   
                // always prefere user input
                MoveAgent(Input.GetAxisRaw("Horizontal"), Input.GetAxisRaw("Vertical"));
            }
            else
            {
                // use the remote move model
                MoveModel moveModel = Status.instance.networkMoveModel;

                MoveAgent(moveModel.GetHorizontalControllerValue(),
                    moveModel.GetVerticalControllerValue());

                Status.instance.networkMoveModel = new MoveModel();

                // allow sending the next frame
                Status.instance.shouldSendImage = true;
            }
        }
        else
        {
            // stop the movement
            StopMovement();
        }
    }

    private bool HasUserInput()
    {
        return Input.anyKey;
    }

    private void ResetAgent()
    {
        this.transform.rotation = initialRotation;
        this.transform.position = initialPosition;

        Status.instance.isOnTrack = true;
        Status.instance.resetAgent = false;
        Status.instance.networkMoveModel = new MoveModel();
    }

    private void StopMovement()
    {
        Status.instance.networkMoveModel = new MoveModel();

        CharacterController characterController = GetComponent<CharacterController>();
        characterController.Move(Vector3.zero);
    }

    private void ApplyGravity()
    {
        CharacterController characterController = GetComponent<CharacterController>();

        Vector3 moveDirection = Vector3.zero;
        moveDirection.y -= gravity * Time.deltaTime;
        moveDirection = this.transform.TransformDirection(moveDirection);
        characterController.Move(moveDirection * Time.deltaTime);
    }

    private void MoveAgent(float horizontal, float vertical)
    {
        // always apply gravity first
        ApplyGravity();

        CharacterController characterController = GetComponent<CharacterController>();

        if (characterController.isGrounded && Status.instance.isOnTrack)
        {
            Vector3 rotation = new Vector3(0, horizontal * rotationSpeed * Time.deltaTime, 0);
            Vector3 move = new Vector3(0, 0, vertical * Time.deltaTime);
            move = this.transform.TransformDirection(move * speed);

            characterController.Move(move);
            this.transform.Rotate(rotation);
        }
        else if(Status.instance.isOnTrack)
        {
            // stop the motion
            StopMovement();
            Status.instance.isOnTrack = false;
        }
    }
}
