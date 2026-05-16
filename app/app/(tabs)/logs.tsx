import { StyleSheet, Text, View } from 'react-native';

export default function LogScreen() {
    return (
        <View style={styles.container}>
            <Text style={styles.text}>To be implemented.</Text>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#25292e',
        justifyContent: 'center',
        alignItems: 'center',
    },
    text: {
        color: '#fff',
    },
});
