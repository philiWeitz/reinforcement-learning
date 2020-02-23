using UnityEngine;

public class AgentMovement : MonoBehaviour
{
    public float speed = 6.0f;

    public float gravity = 20.0f;

    private Vector3 initialPosition;

    private Quaternion initialRotation;

    private Vector3 moveDirection = Vector3.zero;


    void Start()
    {
        initialPosition = this.transform.position;
        initialRotation = this.transform.rotation;
    }

    void Update()
    {
        if (Status.instance.resetAgent)
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
        return Input.GetAxisRaw("Horizontal") < -0.1 || Input.GetAxisRaw("Horizontal") > 0.1 ||
            Input.GetAxisRaw("Vertical") < -0.1 || Input.GetAxisRaw("Vertical") > 0.1;
    }

    private void ResetAgent()
    {
        this.transform.rotation = initialRotation;
        this.transform.position = initialPosition;

        Status.instance.isOnTrack = true;
        Status.instance.resetAgent = false;
        Status.instance.isSimulationRunning = true;
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
            if (characterController.isGrounded)
            {
                // We are grounded, so recalculate move direction directly from axes
                moveDirection = new Vector3(0.0f, 0.0f, vertical);
                moveDirection *= speed;
            }

            Vector3 rotation = new Vector3(0, horizontal * 90 * Time.deltaTime, 0);
            this.transform.Rotate(rotation);

            moveDirection = this.transform.TransformDirection(moveDirection);
            characterController.Move(moveDirection * Time.deltaTime);
        }
        else if(Status.instance.isOnTrack)
        {
            // stop the motion
            StopMovement();
            Status.instance.isOnTrack = false;
        }
    }
}
