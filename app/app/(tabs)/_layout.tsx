import Ionicons from '@expo/vector-icons/Ionicons';
import { Tabs } from 'expo-router';

export default function TabLayout() {
    return (
        <Tabs screenOptions={{
            tabBarActiveTintColor: '#ffd33d',
            headerTintColor: '#fff',
            tabBarStyle: {
                backgroundColor: '#25292e',
            },
        }}>
            <Tabs.Screen name="index" options={{
                title: 'Lens', headerShown: false, tabBarIcon: ({ color, focused }) => (
                    <Ionicons name={focused ? 'camera-sharp' : 'camera-outline'} color={color} size={24} />
                )
            }} />
            <Tabs.Screen name="logs" options={{
                title: 'Logs', headerShown: false, tabBarIcon: ({ color, focused }) => (
                    <Ionicons name={focused ? 'list-sharp' : 'list-outline'} color={color} size={24} />
                )
            }} />
        </Tabs>
    );
}
