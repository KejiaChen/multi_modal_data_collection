#!/usr/bin/env python3
import os
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


INPUT_EVENT_STRUCT = struct.Struct("llHHI")
EV_KEY = 1
KEY_DOWN = 1


class FootswitchTriggerNode(Node):
    def __init__(self):
        super().__init__('footswitch_trigger_node')

        self.declare_parameter(
            'device_path',
            '/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd',
        )
        self.declare_parameter('start_code', 30)
        self.declare_parameter('stop_code', 48)
        self.declare_parameter('custom_code', 46)
        self.declare_parameter('custom_service_name', '')
        self.declare_parameter('debounce_sec', 0.15)

        self.device_path = self.get_parameter('device_path').get_parameter_value().string_value
        self.start_code = self.get_parameter('start_code').get_parameter_value().integer_value
        self.stop_code = self.get_parameter('stop_code').get_parameter_value().integer_value
        self.custom_code = self.get_parameter('custom_code').get_parameter_value().integer_value
        self.custom_service_name = (
            self.get_parameter('custom_service_name').get_parameter_value().string_value.strip()
        )
        self.debounce_sec = (
            self.get_parameter('debounce_sec').get_parameter_value().double_value
        )

        self.device_fd = None
        self.running = True
        self.buffer = b''
        self.last_press_times = {}

        self.start_client = self.create_client(Trigger, '/start_episode')
        self.stop_client = self.create_client(Trigger, '/stop_episode')
        self.custom_client = None

        self._wait_for_service(self.start_client, '/start_episode')
        self._wait_for_service(self.stop_client, '/stop_episode')

        if self.custom_service_name:
            self.custom_client = self.create_client(Trigger, self.custom_service_name)
            self._wait_for_service(self.custom_client, self.custom_service_name)

        self.code_actions = {
            self.start_code: ('start', self.start_client),
            self.stop_code: ('stop', self.stop_client),
        }
        if self.custom_code not in self.code_actions:
            self.code_actions[self.custom_code] = ('custom', self.custom_client)

        self._open_device()

        self.reader_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.reader_thread.start()

        custom_desc = self.custom_service_name or 'log only'
        self.get_logger().info(
            'FootswitchTriggerNode ready. '
            f'device={self.device_path}, '
            f'start_code={self.start_code}, '
            f'stop_code={self.stop_code}, '
            f'custom_code={self.custom_code} ({custom_desc})'
        )

    def _wait_for_service(self, client, service_name):
        while self.running and not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{service_name} not available, waiting...')

    def _open_device(self):
        try:
            self.device_fd = os.open(self.device_path, os.O_RDONLY | os.O_NONBLOCK)
            self.get_logger().info(f'Connected to footswitch input device at {self.device_path}')
        except OSError as exc:
            raise RuntimeError(f'Failed to open footswitch device {self.device_path}: {exc}') from exc

    def _input_loop(self):
        while self.running:
            try:
                chunk = os.read(self.device_fd, 4096)
                if not chunk:
                    time.sleep(0.01)
                    continue
                self.buffer += chunk

                while len(self.buffer) >= INPUT_EVENT_STRUCT.size:
                    event = self.buffer[:INPUT_EVENT_STRUCT.size]
                    self.buffer = self.buffer[INPUT_EVENT_STRUCT.size:]
                    _, _, event_type, code, value = INPUT_EVENT_STRUCT.unpack(event)
                    self._process_event(event_type, code, value)
            except BlockingIOError:
                time.sleep(0.01)
            except OSError as exc:
                if self.running:
                    self.get_logger().warn(f'Input read error: {exc}')
                    time.sleep(0.1)

    def _process_event(self, event_type, code, value):
        if event_type != EV_KEY or value != KEY_DOWN:
            return

        now = time.time()
        last_press = self.last_press_times.get(code, 0.0)
        if now - last_press < self.debounce_sec:
            return
        self.last_press_times[code] = now

        action = self.code_actions.get(code)
        if action is None:
            self.get_logger().info(f'Ignoring unmapped pedal code {code}')
            return

        action_name, client = action
        if action_name == 'custom' and client is None:
            self.get_logger().info(
                f'Custom pedal pressed (code {code}), but no custom_service_name is configured.'
            )
            return

        self.get_logger().info(f'Pedal code {code} pressed -> {action_name}')
        self.call_service_async(client, action_name)

    def call_service_async(self, client, action):
        req = Trigger.Request()
        future = client.call_async(req)

        def done_callback(fut):
            try:
                res = fut.result()
            except Exception as exc:
                self.get_logger().error(f'{action.upper()} failed: {exc}')
                return

            msg = res.message if res.message else '(no message)'
            self.get_logger().info(f'{action.upper()} done: {msg}')

        future.add_done_callback(done_callback)

    def destroy_node(self):
        self.running = False
        time.sleep(0.05)
        if self.device_fd is not None:
            try:
                os.close(self.device_fd)
            except OSError:
                pass
            self.device_fd = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FootswitchTriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
